# Stereo-Visual-Odometry
A Python implementation of a classical stereo visual odoemtry pipeline using the KITTI dataset.

An article I wrote describing this system in greater depth can be found [here]()

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
Go to this [link](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download 3 (+1 optional) datasets from KITTI:
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
Run `python vo.py` to visualize the pipeline on sequence 00 (as configured in main). Alter the sequence value to test other trajectories (00-21 are valid inputs).

## Results + Room for Improvement
Using classical visual odometry results in relatively impressive trajectory estimations considering we are not performing pose graph optimizations or even using LiDAR!
![Sequence_03](/assets/seq_03.png)
![Sequence_09](/assets/seq_09.png)\

### Drift
However, as mentioned earlier, this approach is prone to drift. This is due to our errors compounding over time and not performing any corrective measures (like PGO). This is well demonstrated in sequence 00
![Sequence_00](/assets/seq_00.png)\

### Moving Objects in Frame
Another issue with this relatively simple approach is the assumption that the objects in our images are stationary. This assumption poses a challenge to our system when other cars are driving in the scene (or any object is moving for that matter). An example of this is in sequence 07. At the bottom left of the diagram our car is sitting idle, but cross-traffic drives in front of it. This causes the unintended shift horizontally.
![Sequence_07](/assets/seq_07.png)\


## Credit + Thanks
Much of the code was derived from Nate Cibik and his tutorial series on [YouTube](https://www.youtube.com/watch?v=SXW0CplaTTQ&list=PLrHDCRerOaI9HfgZDbiEncG5dx7S3Nz6X). He does a fantastic job describing the fundamentals of visual odometry and I would highly recommend watching his breakdown if you prefer going step-by-step through a notebook. 