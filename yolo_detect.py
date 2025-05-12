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
