import cv2
import numpy as np
import argparse
import logging
import os

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 矫正图片（把aruco从平行四边形变成正方形需要一个矩阵，M，用M处理原图得到矫正后的图片，保存起来）


class PerspectiveCorrector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = None

    def load_image(self):
        """加载图像"""
        try:
            # 读取图像
            self.img = cv2.imread(self.image_path)
            if self.img is None:
                raise ValueError("无法读取图像，请检查路径是否正确。")
        except Exception as e:
            logging.error(f"图像加载失败: {e}")
            exit(1)

    def detect_and_correct_perspective(self):
        """检测 ArUco 标记并矫正图像"""
        try:
            # 检测 ArUco 标记
            parameters = cv2.aruco.DetectorParameters_create()
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
            corners, ids, _ = cv2.aruco.detectMarkers(self.img, aruco_dict, parameters=parameters)

            if corners and len(corners) >= 1:
                # 使用第一个检测到的 ArUco 标记
                marker_corners = corners[0][0]  # 获取第一个标记的四个角点
                self.correct_perspective(marker_corners)
            else:
                logging.warning("未检测到 ArUco 标记，跳过矫正。")
        except Exception as e:
            logging.error(f"矫正失败: {e}")

    def correct_perspective(self, marker_corners):
        """利用 ArUco 标记矫正整个图像"""
        # 动态计算正方形的边长
        square_size = self.calculate_square_size(marker_corners)

        # 定义目标矩形的四个角点（一个正方形）
        dst_points = np.array([
            [0, 0],                         # 左上角
            [square_size, 0],               # 右上角
            [square_size, square_size],     # 右下角
            [0, square_size]                # 左下角
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(marker_corners.astype(np.float32), dst_points)

        # 获取原图的尺寸
        height, width = self.img.shape[:2]

        # 计算矫正后图像的边界框
        transformed_corners = cv2.perspectiveTransform(
            np.array([[[0, 0], [width, 0], [width, height], [0, height]]], dtype=np.float32),
            M
        )

        # 找到矫正后图像的边界
        min_x = int(np.min(transformed_corners[0][:, 0]))
        max_x = int(np.max(transformed_corners[0][:, 0]))
        min_y = int(np.min(transformed_corners[0][:, 1]))
        max_y = int(np.max(transformed_corners[0][:, 1]))

        # 计算矫正后图像的宽度和高度
        new_width = max_x - min_x
        new_height = max_y - min_y

        # 调整透视变换矩阵以适应新的输出尺寸
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)
        final_transform = translation_matrix @ M

        # 应用透视变换到整个图像
        corrected_img = cv2.warpPerspective(self.img, final_transform, (new_width, new_height))

        # 保存矫正后的图像
        self.save_corrected_image(corrected_img)

    def calculate_square_size(self, marker_corners):
        """
        动态计算正方形的边长。
        通过计算 ArUco 标记的平均边长来确定正方形的尺寸。
        """
        # 提取四个角点
        p1, p2, p3, p4 = marker_corners

        # 计算四条边的长度
        side1 = np.linalg.norm(p1 - p2)  # 边1：p1 到 p2
        side2 = np.linalg.norm(p2 - p3)  # 边2：p2 到 p3
        side3 = np.linalg.norm(p3 - p4)  # 边3：p3 到 p4
        side4 = np.linalg.norm(p4 - p1)  # 边4：p4 到 p1

        # 计算平均边长
        average_side_length = (side1 + side2 + side3 + side4) / 4

        # 返回平均边长作为正方形的边长
        return int(round(average_side_length))

    def save_corrected_image(self, corrected_img):
        """保存矫正后的图像到原始图像路径"""
        try:
            # 获取原始图像路径的相关信息
            dir_name, file_name = os.path.split(self.image_path)
            base_name, ext = os.path.splitext(file_name)

            # 构造新的文件名（加后缀 _corrected）
            new_file_name = f"{base_name}_corrected{ext}"
            new_file_path = os.path.join(dir_name, new_file_name)

            # 保存矫正后的图像
            cv2.imwrite(new_file_path, corrected_img)
            logging.info(f"矫正后的图像已保存为: {new_file_path}")
        except Exception as e:
            logging.error(f"保存矫正后的图像失败: {e}")


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="梯形矫正工具")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    args = parser.parse_args()

    # 初始化矫正工具
    corrector = PerspectiveCorrector(image_path=args.image)

    # 加载图像
    corrector.load_image()

    # 检测 ArUco 标记并矫正图像
    corrector.detect_and_correct_perspective()