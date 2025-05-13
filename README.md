# Currency Detection System with YOLO

![Currency Detection](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2FtZ2NtbjNkZDEzNnVlOG15ZGgxaDltdjFiNG15amliNXVqOHFiNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/nfspjG8dH3TaVuY1CX/giphy.gif)

## Overview

This project implements a real-time currency detection and counting system using YOLO (You Only Look Once) deep learning models. The system can identify and count various currency denominations through a webcam feed, providing an efficient way to count money automatically.

### Key Features

- **Real-time detection** of multiple currency notes simultaneously
- Support for **14 different currency denominations** including both Arabic and English variants
- **Monetary value tracking and summation** of detected notes
- **GPU-accelerated inference** with fallback to CPU
- **Optimized for edge devices** with configurable inference settings
- **User-friendly UI** showing detected denominations and total value


## Currency Denominations Supported

The model is trained to detect the following currency notes:
- 5 (Arabic face & English back)
- 10 (New, Old Arabic face, Old English back)
- 20 (Old Arabic face & Old English back)
- 50 (Arabic face & English back)
- 100 (Arabic face & English back)
- 200 (Arabic face & English back)

## Requirements

- Python 3.8+ 
- PyTorch 1.10+
- Ultralytics
- OpenCV
- CUDA toolkit (optional, for GPU acceleration)

## Installation

```bash
# Clone this repository
git clone https://github.com/loxrobby/Image-Project.git
cd Image-Project

# Install required packages
pip install -r requirements.txt

# Download pre-trained model (if not included in the repo)
# Place my_model.pt in the project root directory
```

## Usage

### Running the Detection System

```bash
python yolo_detect.py
```

### Controls:
- **'q'** - Exit the application
- **'s'** - Pause detection 
- **'p'** - Save current frame as image

### Configuration Options

Edit the following parameters in `yolo_detect.py` to customize the detection system:

```python
# Model and camera settings
model_path = "my_model.pt"  # Path to your trained YOLO model
resW, resH = 640, 480       # Camera resolution
min_thresh = 0.5            # Minimum detection confidence threshold

# Performance optimization settings
inference_size = (320, 240) # Size for inference (smaller = faster, less accurate)
process_every_n_frames = 1  # Process every nth frame (higher = faster, less smooth)
max_objects = 10            # Maximum objects to detect per frame
```

## Training Your Own Model

This repository includes a Jupyter notebook `Train_YOLO_Models.ipynb` that walks through the process of training custom YOLO models using Google Colab. The notebook covers:

1. Preparing your dataset
2. Setting up the training environment
3. Training a YOLO model (YOLOv11, YOLOv8, or YOLOv5)
4. Evaluating model performance
5. Exporting the model for deployment

### Dataset Preparation

The dataset should be organized as follows:
```
custom_data/
├── images/         # All training images
├── labels/         # YOLO format annotation files
└── classes.txt     # List of class names
```

The Egyptian Currency dataset used in this project is available on Roboflow Universe:
[Egyptian Currency Dataset](https://universe.roboflow.com/banha-university-dxs4z/egyptian-currency-psnkr)

## Model Architecture

The project uses the YOLO (You Only Look Once) architecture, specifically supporting YOLOv11s model. YOLO is a state-of-the-art real-time object detection system that is both fast and accurate.

For more information on YOLOv11, please refer to the [official Ultralytics YOLOv11 documentation](https://docs.ultralytics.com/models/yolo11/).

## Project Documentation

Detailed documentation of our Currency Recognition System implementation is available in our project report:
[Currency Recognition System Using YOLOv11s Documentation.pdf](https://drive.google.com/file/d/1LnPM2tlmLWvSfCkBDKIKPyyqpk2lSFvn/view?usp=sharing)

## Performance Optimization

The detection system includes several optimizations to improve performance:
- Dynamic CUDA/CPU selection based on available hardware
- Inference size reduction for faster processing
- Frame skipping option for resource-constrained systems
- Multi-threading support for OpenCV operations
- cuDNN optimizations for GPU acceleration

## Development Roadmap

- [ ] Add support for more currency types
- [ ] Implement currency orientation detection
- [ ] Develop mobile application version
- [ ] Add support for video file input
- [ ] Implement serial counting (tracking notes over time)

## Team Members

- Yousef Salah Nabil (ID: 202202424)
- Yousef Hassan AbdelMoaty (ID: 202202011)
- Yousef Mohamed Abdelkaleq (ID: 202202446)
- Yousef Hussin Ahmed (ID: 202205654)
- Mohamed Yassr Arafat (ID: 202203301)
- Yassin Hussin Meky (ID: 202201988)
- Hisham Mohamed Badawi (ID: 202201817)

## Lab Instructors

- ENG. Ahmed Zakaria
- ENG. Ahmed Ali
- ENG. Ahmed Khaled
- ENG. Mohamed AbdelMoneim

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [EdjeElectronics](https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models) for the training pipeline

## License

[MIT License](LICENSE) - Feel free to use this project for your applications.

## Project Repository

- **GitHub**: [https://github.com/loxrobby/Image-Project](https://github.com/loxrobby/Image-Project)

---

If you find this project useful, please consider giving it a star on GitHub!
