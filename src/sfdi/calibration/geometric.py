import cv2

def undistort_img(img, cam_mat, dist_mat, optimal_mat):
    return cv2.undistort(img, cam_mat, dist_mat, None, optimal_mat)