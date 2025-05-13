import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Enable cuDNN autotuner for optimal GPU performance
torch.backends.cudnn.benchmark = True

# Model and camera settings
model_path = "my_model.pt"
resW, resH = 640, 480
min_thresh = 0.5

# Performance optimization settings
inference_size = (320, 240)  # Smaller size for faster inference
process_every_n_frames = 1
max_objects = 10
mirror_input = True  # Mirror the input since model was trained on mirrored images

# Retention settings for detection results
retention_time = 0.5  # Seconds to keep a denomination after it disappears

# OpenCV optimization
cv2.setUseOptimized(True)
cv2.setNumThreads(4)  # Adjust based on your CPU

# Check if model exists
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(0)

# Display PyTorch version information
print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

# Check for CUDA availability with improved diagnostics
device = 'cpu'  # Default to CPU
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    print(f"CUDA is available. CUDA version: {cuda_version}")
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} CUDA device(s):")
    
    for i in range(gpu_count):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {gpu_props.name}, {gpu_props.total_memory / 1024**3:.2f} GB memory")
    
    # Try to initialize CUDA and handle potential issues
    try:
        # Test CUDA initialization
        test_tensor = torch.zeros(1).cuda()
        device = 'cuda'
        print("GPU initialization successful. Using GPU for inference.")
        
        # Set CUDA device properties for better performance
        torch.cuda.set_device(0)  # Use first GPU
        torch.backends.cudnn.enabled = True
    except RuntimeError as e:
        print(f"WARNING: CUDA initialization failed: {e}")
        print("Falling back to CPU for inference.")
else:
    print("WARNING: CUDA is NOT available. Using CPU for inference.")

# Workaround: force CPU if torchvision NMS CUDA is not available
if device == 'cuda':
    try:
        from torchvision.ops import nms # Import NMS from torchvision Non-Maximum Suppression
        dummy_boxes = torch.rand(1, 4).cuda()
        dummy_scores = torch.rand(1).cuda()
        _ = nms(dummy_boxes, dummy_scores, 0.5)
        print("torchvision NMS CUDA test successful.")
    except Exception as e:
        print(f"torchvision NMS CUDA not available: {e}")
        print("Falling back to CPU for inference.")
        device = 'cpu'

# Load model
print("Loading model...")
model = YOLO(model_path, task='detect')
model.to(device)
labels = model.names
print(f"Model loaded successfully with {len(labels)} classes")
print(f"Using device: {model.device}")

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Custom class mapping
class_map = {
    0: "10 old Arabic face",
    1: "5 English Back",
    2: "50 English back",
    3: "10 old English back",
    4: "10 New",
    5: "10 New", 
    6: "100 Arabic face",
    7: "20 old English back",
    8: "200 Arabic face",
    9: "200 English Back",
    10: "50 Arabic face",
    11: "20 Old Arabic face", 
    12: "5 Arabic face",
    13: "100 English Back" 
}

# Map class index to monetary value
value_map = {
    0: 10, 1: 5, 2: 50, 3: 10, 4: 10, 5: 10,
    6: 100, 7: 20, 8: 200, 9: 200, 10: 50,
    11: 20, 12: 5, 13: 100
}

# Initialize webcam (use DirectShow for faster access on Windows)
print("Opening USB2.0 HD UVC WebCam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increase buffer size

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open webcam. Trying other indices...")
    for i in range(1, 5):
        cap = cv2.VideoCapture(i)
        cap.set(3, resW)
        cap.set(4, resH)
        if cap.isOpened():
            print(f"Found camera at index {i}")
            break
    
    if not cap.isOpened():
        print("Could not open USB2.0 HD UVC WebCam. Please check connection.")
        sys.exit(0)

# Initialize tracking variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 50
frame_count = 0

# Detection retention variables
denomination_last_seen = {}  # Dictionary to store {denomination: timestamp}
active_denominations = {}    # Continues to track the note even if it disappears for a few frames, based on a timer.
current_total = 0            # Current total value to display

print("Starting detection with USB2.0 HD UVC WebCam...")

# Begin inference loop
while True:
    t_start = time.perf_counter() #This records a precise timestamp the start of the current video frame.
    current_time = time.time()  # Get current time for retention calculations seconds since January 1, 1970
    current_frame_denominations = {}  # Shows what was detected in each individual frame.
    
    # Grab frame from camera
    ret, frame = cap.read()
    if (frame is None) or (not ret):
        print('Unable to read frames from the camera.')
        break
    
    frame_count += 1
    
    # Mirror the frame if required
    if mirror_input:
        frame = cv2.flip(frame, 1)
    
    # Process frames
    if frame_count % process_every_n_frames == 0:
        # Resize for inference (smaller = faster)
        input_frame = cv2.resize(frame, inference_size, interpolation=cv2.INTER_LINEAR) #INTER_LINEAR is a good default for resizing real-world images.
        try:
            # Run inference on the selected device
            results = model(input_frame, verbose=False, conf=min_thresh, device=device)
            
            # Extract results
            detections = results[0].boxes #Each detection has coordinates, class index, and confidence.

            object_count = 0

            # Calculate scaling factors (width, height)
            x_scale = frame.shape[1] / inference_size[0]
            y_scale = frame.shape[0] / inference_size[1]
            
            # Process detections
            for i in range(min(len(detections), max_objects)):
                # Get bounding box coordinates
                xyxy_tensor = detections[i].xyxy
                xyxy = xyxy_tensor.detach().cpu().numpy().squeeze()
                
                # Handle single detection case
                if xyxy.ndim == 1:
                    # Correct scaling: multiply x by x_scale, y by y_scale
                    xmin = int(xyxy[0] * x_scale)
                    ymin = int(xyxy[1] * y_scale)
                    xmax = int(xyxy[2] * x_scale)
                    ymax = int(xyxy[3] * y_scale)
                else:
                    continue
                
                # Get class info and confidence
                classidx = int(detections[i].cls.item())
                # Use custom mapping if available, else fallback to model label
                classname = class_map.get(classidx, labels[classidx])
                conf = detections[i].conf.item()
                
                # Draw box if confidence threshold is met
                if conf > min_thresh:
                    color = bbox_colors[classidx % 10]
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
                    
                    # Draw label
                    label = f'{classname}: {int(conf*100)}%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(ymin, labelSize[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Accumulate denominations for current frame
                    value = value_map.get(classidx, 0)
                    if value > 0:
                        current_frame_denominations[value] = current_frame_denominations.get(value, 0) + 1
                    
                    object_count += 1
            
            # Update timestamp for each detected denomination
            for value, count in current_frame_denominations.items():
                denomination_last_seen[value] = current_time
                active_denominations[value] = count  # Update count immediately
            
            # Check for denominations that have disappeared for too long
            denominations_to_remove = []
            for value in list(active_denominations.keys()):
                if value not in current_frame_denominations:
                    # Denomination not seen in current frame
                    last_seen_time = denomination_last_seen.get(value, 0)
                    if current_time - last_seen_time > retention_time:
                        # More than retention_time seconds since we last saw this denomination
                        denominations_to_remove.append(value)
            
            # Remove expired denominations
            for value in denominations_to_remove:
                active_denominations.pop(value, None)
                denomination_last_seen.pop(value, None)
            
            # Calculate current total value
            current_total = sum(value * count for value, count in active_denominations.items())
            
            # Update frame with info
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
            cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

            # Format and display denomination summary in the top-right corner
            summary_lines = []
            # Sort denominations by value for consistent display
            for value in sorted(active_denominations.keys()):
                count = active_denominations[value]
                summary_lines.append(f'{value} LE: {count}')
            summary_lines.append(f'Total: {current_total} L.E.')

            # Display summary lines
            y_offset = 25 # Starting y position for the summary (adjusted for larger font)
            summary_font_scale = 0.7
            summary_thickness = 2
            summary_color = (255, 0, 255) # Magenta color
            line_spacing = 25 # Adjusted spacing for larger font

            for line in summary_lines:
                text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, summary_font_scale, summary_thickness)
                text_x = frame.shape[1] - text_size[0] - 10 # 10 pixels padding from right edge
                cv2.putText(frame, line, (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, summary_font_scale, summary_color, summary_thickness)
                y_offset += line_spacing # Move down for the next line

        except Exception as e:
            print(f"Inference error: {e}")
            cv2.putText(frame, "Error - Check console", (10,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
    
    # Display results
    cv2.imshow('USB2.0 HD UVC WebCam - YOLO Detection', frame)
    
    # Handle user input
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'):  # Press 's' to pause
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture
        cv2.imwrite('capture.png', frame)
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Maintain FPS buffer for averaging
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
cap.release()
cv2.destroyAllWindows()