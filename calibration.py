import numpy as np
import cv2
import os
import argparse

def calibrate(dirpath, square_size, width, height, visualize=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # Ensure the directory exists
    if not os.path.exists(dirpath):
        print(f"指定的目录不存在：{dirpath}")
        return None

    # Get all image files in the directory
    images = [f for f in os.listdir(dirpath) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print(f"目录中没有找到任何图像文件：{dirpath}")
        return None

    for fname in images:
        img_path = os.path.join(dirpath, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图像：{img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            if visualize:
                cv2.imshow('img', img)
                cv2.waitKey(0)
        else:
            print(f"棋盘格角点未检测到：{img_path}")

    cv2.destroyAllWindows()

    if not imgpoints:
        print("没有有效的图像点，无法进行相机标定")
        return None

    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Path to folder containing checkerboard images for calibration")
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=9)", default=9)
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)", default=6)
    ap.add_argument("-s", "--square_size", type=float, default=1, help="Length of one edge (in metres)")
    ap.add_argument("-v", "--visualize", type=str, default="False", help="To visualize each checkerboard image")
    args = vars(ap.parse_args())

    dirpath = args['dir']
    square_size = args['square_size']
    width = args['width']
    height = args['height']

    visualize = args["visualize"].lower() == "true"

    result = calibrate(dirpath, square_size, width, height, visualize=visualize)
    if result:
        ret, mtx, dist, rvecs, tvecs = result
        print("Camera calibration successful")
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        print("Rotation vectors:\n", rvecs)
        print("Translation vectors:\n", tvecs)

        # Create output directory if it doesn't exist
        output_dir = "output/camera_calibration_data"
        os.makedirs(output_dir, exist_ok=True)

        # Save calibration results
        mtx_file = os.path.join(output_dir, "calibration_matrix.npy")
        dist_file = os.path.join(output_dir, "distortion_coefficients.npy")

        np.save(mtx_file, mtx)
        np.save(dist_file, dist)

        print(f"Calibration results saved to: {output_dir}")
    else:
        print("Camera calibration failed")