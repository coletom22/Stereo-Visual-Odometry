# Stereo-Visual-Odometry

Stereo Visual Odometry is the process of understanding the movements of a vehicle based on images taken by cameras mounted to the car. The system involves 4 key steps:
1. Using **Semi-Global Block Matching** (SGBM) to generate a disparity map and calculate depth
2. Extracting features with **Scale-Invariant Feature Transform** (SIFT), filtering with Lowe's Ratio test, and matching with brute force for highest accuracy 
3. Calculating trajectory based on the **Point-n-Perspective Random Sample Consensus** (PnPRANSAC) algorithm