import cv2
import logging
import numpy as np

def undistort_img(img, cam_mat, dist_mat, optimal_mat):
    return cv2.undistort(img, cam_mat, dist_mat, None, optimal_mat)

def camera_calibration(imgs):
    logger = logging.getLogger("sfdi")
    
    if len(imgs) < 10:
        logger.error(f"10 or more images are required to calibrate a camera ({len(imgs)} provided)")
        return None
    
    logger.info(f"Using {len(imgs)} checkerboard images to calibrate the camera")

    CHECKERBOARD = (7, 5)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    threedpoints = []
    twodpoints = []

    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    for i, image in enumerate(imgs):
        #image = np.array(256 - image, dtype=np.uint8)
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(grey_img, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            threedpoints.append(objectp3d)

            corners2 = cv2.cornerSubPix(grey_img, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)
            
            #image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
        else:
            logger.warning(f'Failed to find checkerboard on image {i}')

    h, w = imgs[0].shape[:2]

    ret, cam_mat, dist_mat, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grey_img.shape[::-1], None, None)

    if not ret: raise

    optimal_mat, roi = cv2.getOptimalNewCameraMatrix(cam_mat, dist_mat, (w, h), 1, (w, h))

    print(f'Camera matrix: {cam_mat}')
    print(f'Dist matrix: {dist_mat}')
    print(f'Optimal matrix: {optimal_mat}')
    
    return cam_mat, dist_mat, optimal_mat