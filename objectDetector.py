import cv2
import numpy as np

class objDetector():
    def __init__(self):
        pass

    def detect_object(self, image):
        # 转换为灰度图
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 自适应阈值处理
        mask = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)

        # 形态学操作去噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_contours = []
        min_area = image.shape[0] * image.shape[1] * 0.001  # 设定最小面积阈值

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                object_contours.append(cnt)

        return object_contours
