import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import os


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    is_camera_facing_aruco - Whether the camera is facing the ArUco marker
    '''

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 加载 ArUco 字典并创建检测参数
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    # 检测 ArUco 标记
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    is_camera_facing_aruco = False  # 是否正对 ArUco 标记

    # 如果检测到标记
    if len(corners) > 0:
        for i in range(len(ids)):
            # 估计每个标记的姿态
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.02, matrix_coefficients, distortion_coefficients
            )

            # 将旋转向量转换为旋转矩阵
            rmat, _ = cv2.Rodrigues(rvec)

            # 计算相机 Z 轴方向在 ArUco 坐标系中的单位向量
            camera_z = np.array([0, 0, 1])  # 相机 Z 轴方向在相机坐标系中
            camera_z_in_aruco = np.dot(rmat.T, camera_z)  # 通过旋转矩阵的逆（转置）进行转换
            camera_z_in_aruco /= np.linalg.norm(camera_z_in_aruco)  # 归一化为单位向量

            # 判断相机是否正对 ArUco 标记
            x, y, z = camera_z_in_aruco
            threshold = 0.1  # 阈值
            if abs(x) < threshold and abs(y) < threshold and abs(z + 1) < threshold:
                is_camera_facing_aruco = True

            # 绘制标记边界框
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # 绘制坐标轴
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, is_camera_facing_aruco


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # 创建输出文件夹
    output_dir = "output/vertical_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用摄像头
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("无法打开摄像头设备，请检查连接或设备索引！")
        sys.exit(1)

    time.sleep(2.0)  # 等待摄像头稳定

    saved_frames = 0  # 已保存的帧数
    max_saved_frames = 5  # 每个位置最多保存的帧数

    while True:
        ret, frame = video.read()

        if not ret:
            print("无法读取摄像头帧，请检查摄像头是否正常工作！")
            break

        # 进行姿态估计
        output, is_camera_facing_aruco = pose_estimation(frame, aruco_dict_type, k, d)

        # 显示结果
        cv2.imshow('Estimated Pose', output)

        # 实时输出相机是否正对 ArUco 标记
        if is_camera_facing_aruco:
            print("Camera is facing the ArUco marker!")

            # 如果满足条件且未达到最大保存帧数，保存当前帧
            if saved_frames < max_saved_frames:
                save_path = os.path.join(output_dir, f"frame_{saved_frames + 1}.png")
                cv2.imwrite(save_path, frame)
                print(f"Saved frame to {save_path}")
                saved_frames += 1
        else:
            print("Camera is NOT facing the ArUco marker.")

        # 按下 ESC 键退出程序
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 键的 ASCII 码是 27
            break

    video.release()
    cv2.destroyAllWindows()