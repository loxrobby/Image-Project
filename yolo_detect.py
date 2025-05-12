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
print(f"PyTorch version: {torch._version_}")
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
        from torchvision.ops import nms
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
active_denominations = {}    # Currently active denominations to display
current_total = 0            # Current total value to display

print("Starting detection with USB2.0 HD UVC WebCam...")
