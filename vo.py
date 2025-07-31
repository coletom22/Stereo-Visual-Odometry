'''
This file is for practicing the fundamentals of visual odometry with the KITTI dataset

Classes:
 - Data_Handler: This class allows us to easily load/access all the necessary data from the KITTI dataset
 
Functions:
 - Decompose projection matrix: shortcut for cv2.decomposeProjectionMatrix because we need to divide out the Z value (t[3]) from the translation vector
    * Arguments:
     @ P: projection matrix
    * Sub functions:
     $ NONE

 - Stereo-to-depth: takes a stereo pair of images and returns a depth map for the left camera
    * Arguments:
     @ img_left: left camera image
     @ img_right: right camera image
     @ P0: left camera projection matrix
     @ P1: right camera projection matrix
     @ matcher: bm or sgbm for matching algorithm
     @ rectified: true by default (kitti is rectified)
    * Sub functions:
     $ Compute left disparity map
     $ Decompose projection matrix
     $ Calculate depth map

 - Estimate Motion: calculate the motion from a pair of subsequent image frames
    * Arguments:
     @ match: list of matched features from the pair of image
     @ kp1: list of keypoints in first image
     @ kp2: list of keypoints in second image
     @ k: camera intrinsic calibration matrix
    * Sub functions:
     $ NONE

 - Pointcloud-to-image: takes a pointcloud of shape Nx4 and projects it onto an image plane, transforming the X, Y, Z coordinates of points
                        to the camera frame with transformation matrix Tr, then projecting them using camera projection matrix P0
    * Arguments:
     @ pointcloud: array of shape Nx4 containing (X, Y, Z, reflectivity)
     @ imheight: height of image plane (pixels)
     @ imwidth: width of image plane
     @ Tr: 3x4 transformation matrix between lidar (X, Y, Z, 1) homogeneous and camera (X, Y, Z)
     @ P0: Projection matrix of camera (should have identity transformation if Tr used)
    * Sub functions:
     $ NONE

 - Extract features: finds keypoints and descriptors for an image using various detector methods
    * Arguments:
     @ image: grayscale image
     @ detector: type of detector (sift or orb)
    * Sub functions:
     $ NONE
    
 - Match features: matches features based on the descriptors from two images
    * Arguments:
     @ des0: list of keypoint descriptors from first image
     @ des1: list of keypoint descriptors from second image
     @ matching: type of matching algorithm (FLANN or BF)
     @ detector: type of detector (sift or orb)
     @ sort: bool on whether or not to sort matches by distance
     @ k: number of neighbors to match for each feature
    * Sub functions:
     $ NONE

 - Filter matches: Use the Lowe ratio test as a threshold to determine if a feature is too ambiguous to keep
    * Arguments:
     @ matches: list of matched features from two images
     @ dist_threshold: max allowed relative distance between the best matches 
    * Sub functions:
     $ NONE

- Compute left disparity map: takes a left and right stereo pair of images and computes the disparity map for the LEFT image
    * Arguments:
     @ img_left: left camera image
     @ img_right: right camera image
     @ matcher: matching algorithm (bm or sgbm)
    * Sub functions:
     $ NONE

- Calculate depth: calculate depth map using a disparity map, intrinsic matrix, translation vectors from camera extrinsic matrices (baseline b)
    * Arguments:
     @ disp_left: disparity map calculated from Compute left disparity map function
     @ k_left: left camera intrinsic matrix
     @ t_left: translation vector from left camera
     @ t_right: translation vector from right camera
     @ rectified: boolean for rectified flag (kitti is rectified)
     
 '''

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

import datetime

#########################################
############ DATA HANDLER ###############
#########################################

class Data_Handler:
    def __init__(self, sequence, lidar=True, low_memory=True):
        self.lidar = lidar
        self.low_memory=low_memory

        self.seq_dir = f'../dataset/sequences/{sequence}/'
        self.pose_file = f'../dataset/poses/{sequence}.txt'

        self.left_image_files = os.listdir(self.seq_dir + 'image_0/')
        self.right_image_files = os.listdir(self.seq_dir + 'image_1/')
        self.velodyne = os.listdir(self.seq_dir + 'velodyne/')
        self.num_frames = len(self.left_image_files)
        self.lidar_path = self.seq_dir + 'velodyne/'

        poses = pd.read_csv(self.pose_file, delimiter=' ', header=None)
        self.gt = np.zeros((len(poses), 3, 4))

        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3,4))

        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3, 4))
        self.P1 = np.array(calib.loc['P1:']).reshape((3, 4))
        self.P2 = np.array(calib.loc['P2:']).reshape((3, 4))
        self.P3 = np.array(calib.loc['P3:']).reshape((3, 4))
        self.Tr = np.array(calib.loc['Tr:']).reshape((3, 4))
        if self.low_memory:
            self.reset_frames()

            self.first_left_image = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[0], 0)
            self.first_right_image = cv2.imread(self.seq_dir + 'image_1/' + self.right_image_files[0], 0)
            self.second_left_image = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[1], 0)
            self.second_right_image = cv2.imread(self.seq_dir + 'image_1/' + self.right_image_files[1], 0)

            if self.lidar:
                self.first_point_cloud = np.fromfile(self.lidar_path + self.velodyne[0], dtype=np.float32, count=-1).reshape((-1, 4))

        self.imwidth = self.first_left_image.shape[1]
        self.imheight = self.first_left_image.shape[0]

    def reset_frames(self):
        self.image_left = (cv2.imread(self.seq_dir + 'image_0/' + left_img, 0)
                           for left_img in self.left_image_files)
        self.image_right = (cv2.imread(self.seq_dir + 'image_1/' + right_img, 0)
                            for right_img in self.right_image_files)
        if self.lidar:
            self.point_clouds = (np.fromfile(self.lidar_path + pt_cld, 
                                             dtype=np.float32, 
                                             count=-1).reshape((-1, 4))
                                for pt_cld in self.velodyne)
        pass




#########################################
########## VISUAL ODOMETRY ##############
#########################################

class Visual_Odometry:
    def __init__(self, dh: Data_Handler):
        self.dh = dh
        self.mask = np.zeros(dh.first_left_image.shape[:2], dtype=np.uint8)
        ymax = self.dh.first_left_image.shape[0]
        xmax = self.dh.first_left_image.shape[1]
        cv2.rectangle(self.mask, (96, 0), (xmax, ymax), (255), thickness=-1) # masking the section of our left image that our right camera does not pick up
        self.trajectory = self.visual_odometry(handler=self.dh,
                             detector='sift',
                             matching='bf',
                             filter_match_distance=0.45,
                             stereo_matcher='sgbm',
                             mask=self.mask,
                             depth_type='stereo',
                             subset=None,
                             plot=True
                             )
    
    def visual_odometry(self, handler, 
                        detector='sift', matching='bf', 
                        filter_match_distance=None, stereo_matcher='bm', 
                        mask=None, depth_type='stereo', 
                        subset=None, plot=False):
        '''
        function to perform visual odometry on a sequence of images from KITTI visual odometry dataset
        '''
        lidar = self.dh.lidar

        # report methods
        print(f'Calculating disparities with stereo {str.upper(stereo_matcher)}')
        print(f'Detecting features with {str.upper(detector)} and watching with {str.upper(matching)}')

        if filter_match_distance is not None:
            print(f'Filtering feature matches with a threshold of {filter_match_distance} * distance')
        if lidar:
            print(f'Improving stereo depth estimate with lidar data')
        if subset is not None:
            num_frames = subset
        else:
            num_frames = self.dh.num_frames
        
        if plot:
            fig = plt.figure(figsize=(14,14))
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=-20, azim=270)
            xs = self.dh.gt[:, 0, 3] # all frames first row, fourth column
            ys = self.dh.gt[:, 1, 3]
            zs = self.dh.gt[:, 2, 3]
            ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
            ax.plot(xs, ys, zs, c='k')

        # establish homogeneous transformation matrix. first pose is Identity matrix
        T_tot = np.eye(4)
        trajectory = np.zeros((num_frames, 3, 4))
        trajectory[0] = T_tot[:3, :] # first 3 rows (not storing homogeneous transformation matrices)

        imheight = self.dh.imheight
        imwidth = self.dh.imwidth

        # decomp left cam projection matrix to get intrinsic mat
        k_left, _, _ = self.decomp_proj_mat(self.dh.P0)

        if self.dh.low_memory:
            self.dh.reset_frames()
            nxt_img = next(self.dh.image_left)
        
        start = datetime.datetime.now()

        # iterate through all frames of the sequence
        for i in range(num_frames-1):
            if i % 250 == 0:
                print(f'Frame {i} completed after {datetime.datetime.now()-start}')
            # get stereo images for depth estimation
            if self.dh.low_memory:
                image_left = nxt_img
                image_right = next(self.dh.image_right)
                nxt_img = next(self.dh.image_left)
            
            if depth_type == 'stereo':
                depth = self.stereo_2_depth(img_left = image_left,
                                            img_right = image_right,
                                            P0 = self.dh.P0,
                                            P1 = self.dh.P1,
                                            matcher = stereo_matcher)
            else:
                depth = None

            if lidar:
                if self.dh.low_memory:
                    pointcloud = next(self.dh.point_clouds)
                lidar_depth = self.pointcloud_2_img(pointcloud=pointcloud,
                                                    imheight=imheight,
                                                    imwidth=imwidth,
                                                    Tr=self.dh.Tr,
                                                    P0=self.dh.P0)
                # find relevant point cloud points (lower distances)
                indices = np.where(lidar_depth < 3000)
                depth[indices] = lidar_depth[indices]

            # get keypoints and descriptors for left camera of two sequential images
            kp1, des1 = self.extract_features(image_left, detector, mask)
            kp2, des2 = self.extract_features(nxt_img, detector, mask)

            # get matches between features in btoh images
            matches_unfiltered = self.match_features(des1,
                                                     des2,
                                                     matching=matching,
                                                     detector=detector,
                                                     sort=False,
                                                     k=2)
            
            # filter matches if distance threshold
            if filter_match_distance is not None:
                matches = self.filter_matches(matches_unfiltered, filter_match_distance)
            else:
                matches = matches_unfiltered
            
            # estimate motion between sequential images of the left camera
            rmat, tvec, _, _ = self.estimate_motion(matches=matches,
                                                    kp1=kp1,
                                                    kp2=kp2,
                                                    k=k_left,
                                                    depth=depth)
            
            # create blank homogeneous transformation matrix
            Tmat = np.eye(4)

            # place resulting rotation mat and translation vector inside
            Tmat[:3, :3] = rmat
            Tmat[:3, 3] = tvec.T

            # two homogeneous transformation matrices dotted together gives the combined transformation of the two matrices
            # i dont think you need to invert if we solve pnp the opposite way (use image 2 points as the 3D)
            T_tot = T_tot.dot(np.linalg.inv(Tmat))

            # place pose estimate in i+1 to correspond to the second image
            trajectory[i+1, :, :] = T_tot[:3, :]

            if plot:
                xs = trajectory[:i+2, 0, 3]
                ys = trajectory[:i+2, 1, 3]
                zs = trajectory[:i+2, 2, 3]
                plt.plot(xs, ys, zs, c='chartreuse')
                plt.pause(1e-32)
        
        end = datetime.datetime.now()

        if plot:
            print('closing')
            plt.pause(100)

        print(f'Total time to compute trajectory: {end-start}')
        return trajectory
        
    def decomp_proj_mat(self, P):
        k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        t = t / t[3]
        return k, r, t

    def compute_disparity_map(self, img_left, img_right, matcher='sgbm'):
        '''
        Accepts a left and right image (from the same point in time) and computes the disparity map for the LEFT image

        SGBM - semi global block matching - works by creating blocks/windows of pixels and sliding over each image.
        Using the left image as reference, the block slides over the right image at the same y (because stereo) and
        looks for a matching block. The pixel distance between the reference block and the found block is the disparity
        value.

        '''

        # the numbers derived here come from opencv's doc on sgbm
        sad_window = 6 # sum of absolute differences
        num_disparities = sad_window*16
        block_size = 11
        matcher_name = matcher

        if matcher_name == 'bm':
            matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                          blockSize=block_size)
        elif matcher_name == 'sgbm':
            matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                           minDisparity=0,
                                           blockSize=block_size,
                                           P1 = 8 * 1 * block_size ** 2,
                                           P2 = 32 * 1 * block_size ** 2,
                                           mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16 # casted as float 32 and fivided by 16 because they converted it to integer
                                                                               # math to decrease compute time. reverting to values it needs to be

        return disp_left

    def calc_depth(self, disp_left, k_left, t_left, t_right):
        '''
        calculate the depth map based on the disparity map
        
        [---------------------Z----------------------------]
           F
        ***|*********************************************** ^
        *  xl                                             * |
             *                                            * |
                  *                                       * |
                       *                                  * |
           F                *                             * |
        ***|*********************************************** X  ^
        *  xr                         *                   * |  |
                    *                       *             * |  X-b
                                *                *        * |  |
                                          *            *  * |  |
                                                      *   O _  _

        Understanding similar triangles we cant create the equivalancies
        1. Z / f = X / xl
        2. Z / f = X-b / xr

        Rearrange both
        Zxl = fX

        Zxr = fX - fb

        Substitute fX into #2
        Zxr = Zxl - fb
        fb = Zxl -Zxr
        fb / (xl-xr) = Z
        
        Our disparity forumla tells us xl - xr = d (the difference in pixel location from left to right is disparity)

        Therefore 
        fb / d = Z

        Z - depth
        f - focal length (in intrinsic mat)
        b - baseline (in extrinsic vec)
        d - disparity
        '''
    
        b = t_right[0]-t_left[0]

        f = k_left[0][0]
        
        # avoid dividing by 0
        disp_left[disp_left == 0.0] = 0.1
        disp_left[disp_left == -1.0] = 0.1

        depth = np.zeros(disp_left.shape)
        depth = f * b / disp_left

        return depth
    
    def stereo_2_depth(self, img_left, img_right, P0, P1, matcher='sgbm'):
        '''
        Takes a stereo pair of images and returns a depth map for the left camera.

        Combines a few functions we already developed
        '''

        disp = self.compute_disparity_map(img_left, img_right, matcher)

        k_left, _, t_left = self.decomp_proj_mat(P0)
        _, _, t_right = self.decomp_proj_mat(P1)

        depth = self.calc_depth(disp, k_left, t_left, t_right)

        return depth
    
    def pointcloud_2_img(self, pointcloud, imheight, imwidth, Tr, P0):
        '''
        Takes a pointcloud shape Nx4 and projects it onto an image plane, first transforming X, Y, Z coordinates of points to the camera frame with transformation
        matrix Tr, then projecting them using camera projection matrix P0
        '''

        # We know the lidar X axis points forward, we need nothing behind the lidar, so we
        # ignore anything with a X value less than or equal to zero
        pointcloud = pointcloud[pointcloud[:, 0] > 0]
        
        # We do not need reflectance info, so drop last column and replace with ones to make
        # coordinates homogeneous for tranformation into the camera coordinate frame
        pointcloud = np.hstack([pointcloud[:, :3], np.ones(pointcloud.shape[0]).reshape((-1,1))])
        
        # Transform pointcloud into camera coordinate frame
        cam_xyz = Tr.dot(pointcloud.T)
        
        # Ignore any points behind the camera (probably redundant but just in case)
        cam_xyz = cam_xyz[:, cam_xyz[2] > 0]
        
        # extract the Z row which is the depth from camera
        depth = cam_xyz[2].copy()
        
        # Project coordinates in camera frame to flat plane at Z=1 by dividing by Z
        cam_xyz /= cam_xyz[2]
        
        # add row of ones to make our 3D coordinates on plane homogeneous for dotting with P0
        cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])
        
        # Get pixel coordinates of X, Y, Z points in camera coordinate frame
        projection = P0.dot(cam_xyz)
        #projection = (projection / projection[2])
        
        # Turn pixels into integers for indexing
        pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')
        #pixel_coordinates = np.array(pixel_coordinates)
        
        # Limit pixel coordinates considered to those that fit on the image plane
        indices = np.where((pixel_coordinates[:, 0] < imwidth)
                        & (pixel_coordinates[:, 0] >= 0)
                        & (pixel_coordinates[:, 1] < imheight)
                        & (pixel_coordinates[:, 1] >= 0)
                        )
        pixel_coordinates = pixel_coordinates[indices]
        depth = depth[indices]
        
        # Establish empty render image, then fill with the depths of each point
        render = np.zeros((imheight, imwidth))
        for j, (u, v) in enumerate(pixel_coordinates):
            if u >= imwidth or u < 0:
                continue
            if v >= imheight or v < 0:
                continue
            render[v, u] = depth[j]
        # Fill zero values with large distance so they will be ignored. (Using same max value)
        render[render == 0.0] = 3861.45
        
        return render
    
    def extract_features(self, image, detector='sift', mask=None):
        '''
        Find keypoints and descriptors for an image
        '''

        if detector == 'sift':
            det = cv2.SIFT_create()
        elif detector == 'orb':
            det = cv2.ORB_create()

        kp, des = det.detectAndCompute(image, mask)

        return kp, des
    
    def match_features(self, des1, des2, matching='bf', detector='sift', sort=True, k=2):
        '''
        Match features from two images
        '''
        if matching == 'bf':
            if detector == 'sift':
                matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
            elif detector == 'orb':
                matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
        elif matching == 'flann':
            KDTREE = 1
            index_params = dict(algorithm=KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        
        matches = matcher.knnMatch(des1, des2, k=k)
        if sort:
            matches = sorted(matches, key=lambda x: x[0].distance)
        
        return matches
    
    def filter_matches(self, matches, dist_threshold):
        '''
        filter matched features from two images by comparing the distance between the second nearest neighbor
        if the second neighbor is relatively close, then our match was AMBIGUOUS and we should ignore
        '''

        filtered_matches = []
        for m, n in matches:
            if m.distance <= dist_threshold * n.distance:
                filtered_matches.append(m)
        
        return filtered_matches
    
    def estimate_motion(self, matches, kp1, kp2, k, depth=None, max_depth=3000):
        '''
        Estimate camera motion from a pair of subsequent image frames
        '''
        rmat = np.eye(3)
        tvec = np.zeros((3,1))
        image1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
        image2_points = np.float32([kp2[m.trainIdx].pt for m in matches])


        # project our 2D points back into 3D space
        if depth is not None:
            cx = k[0, 2] # x center
            cy = k[1, 2] # y center
            fx = k[0, 0] # x focal length
            fy = k[1, 1] # y focal length
            object_points = np.zeros((0,3)) # 3D locations of the features tracked
            delete = []
            # extract depth information of query image at match points and build 3D positions
            for i, (u, v) in enumerate(image1_points):
                z = depth[int(v), int(u)]  # the whole point of stereo vs monocular is to retrieve this z value
                
                # if the depth at the position of our matched features is greater than our max, then
                # we can ignore this feature because we dont actualy know the depth and it will
                # mess up the calculations. We add its index to a list of coordinates to delete from our keypoints
                # lists and continue the loop
                if z > max_depth:
                    delete.append(i)
                    continue

                # use arithmetic to extract x and y
                # subtracting cx and cy from our (u, v) will bring the origin back to the center
                # divide by the focal length to get rid of pixel units (now unitless)
                # multiply by depth at that point (meters)
                x = z*(u-cx)/fx
                y = z*(v-cy)/fy

                object_points = np.vstack([object_points, np.array([x, y, z])])
                # equivalent math with dot product with inverse of k matrix, but slower
                # object_points = np.vstack([object_points, np.linalg.inv(k).dot(n*np.array([u, v, 1]))])

            image1_points = np.delete(image1_points, delete, 0)
            image2_points = np.delete(image2_points, delete, 0)

            # use PnP algorithm with RANSAC for robustness to outliers
            _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
            # Perspective N Point algorithm to solve for transformation. Explains where the 3D points end up in the second image
            # returns axis angle rotation representation rvec, use Rodrigues formula to convert this to our desired format of 3x3 rmat
            rmat = cv2.Rodrigues(rvec)[0]        
        
        else:
            # no depth info provided, use essential matrix decomp instead, this is not really useful
            # since we will get a 3D motion tracking but the scale will be ambigious
            image1_points_hom = np.hstack([image1_points, np.ones(len(image1_points)).reshape(-1, 1)])
            image2_points_hom = np.hstack([image2_points, np.ones(len(image2_points)).reshape(-1, 1)])
            E = cv2.findEssentialMat(image1_points, image2_points, k)[0]
            _, rmat, tvec, mask = cv2.recoverPose(E, image1_points, image2_points, k)

        return rmat, tvec, image1_points, image2_points
    

def calculate_error(ground_truth, estimated, error_type='mse'):
    '''
    Takes an array of ground truth and estimated poses of shape Nx3x4 and comptues error using Euclidean
    distance between the two 3D coordinates at each position
    '''

    num_frames_est = estimated.shape[0]

    def get_mse(gt, est):
        se = np.sqrt((gt[:num_frames_est, 0, 3] - est[:, 0, 3])**2 +
                     (gt[:num_frames_est, 1, 3] - est[:, 1, 3])**2 +
                     (gt[:num_frames_est, 2, 3] - est[:, 2, 3])**2)**2
        mse = se.mean()
        return round(mse, 4)
    
    def get_mae(gt, est):
        ae = np.sqrt((gt[:num_frames_est, 0 ,3] - est[:, 0, 3])**2 +
                     (gt[:num_frames_est, 1, 3] - est[:, 1, 3])**2 +
                     (gt[:num_frames_est, 2, 3] - est[:, 2, 3])**2)
        mae = ae.mean()
        return round(mae, 4)

    if error_type == 'mse':
        return get_mse(ground_truth, estimated)
    elif error_type == 'mae':
        return get_mae(ground_truth, estimated)
    elif error_type == 'rsme':
        return np.sqrt(get_mse(ground_truth, estimated))
    elif error_type == 'all':
        return dict(
            mae = get_mae(ground_truth, estimated),
            mse = get_mse(ground_truth, estimated),
            rsme = np.sqrt(get_mse(ground_truth, estimated))
        )
    
if __name__ == '__main__':
    dh = Data_Handler(sequence='00', lidar=False, low_memory=True)
    vo = Visual_Odometry(dh)
    print(calculate_error(vo.dh.gt, vo.trajectory, 'all'))
