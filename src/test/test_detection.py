import cv2

from pathlib import Path
from opensfdi import image

def test_main():
    img_path = Path("D:\\results\\circles\\calibration\\")

    phases = 12
    stripe_counts = [1.0, 16.0, 64.0]
    poses = 23
    N = poses * len(stripe_counts) * phases * 2

    imgs = [image.FileImage(img_path / f"calibration_{str(i).zfill(4)}.tif", True) for i in range(N)]

    i = 0
    while len(imgs) != 0:

        xs = [img.rawData for img in imgs[:12]]

        dc_img = image.DC(xs)

        # Convert to 0 / 1
        dc_img = image.ThresholdMask(dc_img)

        success = detect_blobs(dc_img, (11, 8))

        if not success:
            print(f"Could not find POIs for image {i}")

        i += 1
        imgs = imgs[12:]

def detect_blobs(img, shape):
    img = image.ToU8(img)

    # Setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()
    blobParams.blobColor = 255

    blobParams.filterByArea = True
    blobParams.maxArea = 2500

    # # Filter by Circularity
    # blobParams.filterByCircularity = True
    # blobParams.minCircularity = 0.1

    # # Filter by Convexity
    # blobParams.filterByConvexity = True
    # blobParams.minConvexity = 0.87

    # # Filter by Inertia
    # blobParams.filterByInertia = True
    # blobParams.minInertiaRatio = 0.01

    # Create a detector with the parameters

    detector = cv2.SimpleBlobDetector_create(blobParams)

    # Check number of blobs detected is the correct amount
    # blobs = detector.detect(img)
    # print(f"Number of blobs: {len(blobs)}")

    result, corners = cv2.findCirclesGrid(img, shape, None,
        flags=cv2.CALIB_CB_SYMMETRIC_GRID,
        blobDetector=detector)

    if not result:
        return False

    cb_corners = cv2.drawChessboardCorners(img, shape, corners, result)

    # image.show_image(cb_corners, size=(1024, 768))

    return True