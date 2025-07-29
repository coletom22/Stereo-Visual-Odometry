# Stereo-Visual-Odometry
A Python implementation of a classical stereo visual odoemtry pipeline using the KITTI dataset.

An article I wrote describing this system in greater depth can be found [here] ()

Stereo Visual Odometry is the process of understanding the movements of a vehicle based on images taken by cameras mounted to the car. The system involves 3 key steps:
1. Using stereo pairs of images to calculate depth
2. Tracking features through subsequent frames
3. Calculating the transformation from one image to the next based on the depth info + features

## Features
- Implementation of stereo visual odometry
- Includes `Data_Handler` class for easy integration with KITTI dataset
- Utilizes OpenCV for image processing, feature extraction, and motion estimation
- Employs **Semi-Global Block Matching** (SGBM) for depth estimation
- Applies **Scale-Invariant Feature Transform** (SIFT) for robust feature detection
- Estimates camera pose using **Perspective-n-Point** (PnP) with **Random Sample Consensus** (RANSAC)
- Visualizes estimated trajectories against the ground truth
- Calculates error metrics (MSE, RMSE, and MAE)

## Setup and Installation
### 1: Clone this Repo
`git clone https://github.com/coletom22/Stereo-Visual-Odometry.git`
### 2: Install dependencies
Run the following in a terminal with your virtual environment active
`pip install -r requirements.txt`
### 3: Download the KITTI datset
Go to this [link] (https://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download 3 (+1 optional) datasets from KITTI:
1. Grayscale (22GB)
2. Calibration files (1MB)
3. Ground truth poses (4MB)
4. Velodyne (80GB) \[optional\]

File structure to be compatible with `Data_Handler` class:
```
├── dataset/
│   ├── sequences/
│   │   ├── 00/
│   │   │   ├── image_0/
│   │   │   ├── image_1/
│   │   │   └── calib.txt
│   │   └── 01/...
│   ├── poses/
│       ├── 00.txt
│       └── 01.txt...
└── Stereo_Visual_Odometry/
    ├── vo.py
    ├── requirements.txt
    └── venv/
```
## Usage
Run
`python vo.py`
to visualize the pipeline on sequence 00 (as configured in main). Alter the sequence value to test other trajectories (00-21 are valid inputs)

## Results + Room for Improvement
Using classical visual odometry results in relatively impressive trajectory estimations considering we are not performing pose graph optimizations or even using LiDAR!
[Sequence_03] (/assets/seq_00.png)
[Sequence_09] (/assets/seq_09.png)
