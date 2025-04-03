import cv2
from objectDetector import objDetector
import numpy as np
import argparse

# Argument parsing for side length input
parser = argparse.ArgumentParser(description="Measure")
parser.add_argument("--side", type=float, required=True, help="Side length of the aruco marker in cm")
args = parser.parse_args()

# call object
detector = objDetector()

# import image data and read it
image_path = 'images/image8.jpg'
img = cv2.imread(image_path)

# Load aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

# Get Aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Draw polygon around the marker
int_corners = np.asarray(corners, dtype=np.int32)  
cv2.polylines(img, int_corners, True, (0, 0, 255), 5)

if corners:
    # Aruco Perimeter
    aruco_perimeter = cv2.arcLength(corners[0], True)

    # Pixel to cm ratio (using multiplication)
    pixel_cm_ratio = (4 * args.side) / aruco_perimeter
else:
    pixel_cm_ratio = 29.526773834228514

# Store clicked points and the state of resizing
points = []
dragging = False
start_point = None
end_point = None

# Store line points for right-click measurement
line_points = []

# Mouse callback function
def click_and_measure(event, x, y, flags, param):
    global points, dragging, start_point, end_point, img, line_points

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        dragging = True
        points.append(start_point)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        end_point = (x, y)
        img_copy = img.copy()
        cv2.rectangle(img_copy, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Measure Object Size", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        end_point = (x, y)
        if len(points) == 1:
            cv2.rectangle(img, start_point, end_point, (0, 255, 0), 8)

            width_pixels = abs(end_point[0] - start_point[0])
            height_pixels = abs(end_point[1] - start_point[1])

            width_cm = width_pixels * pixel_cm_ratio
            height_cm = height_pixels * pixel_cm_ratio

            mid_point = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3  
            thickness = 8  
            shadow_thickness = 12  

            text_w = f'Width: {round(width_cm, 2)} cm'
            text_h = f'Height: {round(height_cm, 2)} cm'

            text_position_w = (mid_point[0] - 200, mid_point[1] - 50)
            text_position_h = (mid_point[0] - 200, mid_point[1] + 50)

            cv2.putText(img, text_w, text_position_w, font, font_scale, (0, 0, 0), shadow_thickness)
            cv2.putText(img, text_h, text_position_h, font, font_scale, (0, 0, 0), shadow_thickness)

            cv2.putText(img, text_w, text_position_w, font, font_scale, (0, 255, 255), thickness)
            cv2.putText(img, text_h, text_position_h, font, font_scale, (0, 255, 255), thickness)

            points = []
            cv2.imshow("Measure Object Size", img)

    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击选择两点连线并测量距离
        if len(line_points) < 2:
            line_points.append((x, y))

        if len(line_points) == 2:
            cv2.line(img, line_points[0], line_points[1], (255, 0, 0), 5)  # 画线
            dist_pixels = np.linalg.norm(np.array(line_points[0]) - np.array(line_points[1]))
            dist_cm = dist_pixels * pixel_cm_ratio

            mid_point = ((line_points[0][0] + line_points[1][0]) // 2, (line_points[0][1] + line_points[1][1]) // 2)

            text_d = f'Distance: {round(dist_cm, 2)} cm'

            cv2.putText(img, text_d, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)
            cv2.putText(img, text_d, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

            line_points = []  # 重置线段点
            cv2.imshow("Measure Object Size", img)

# 创建一个可缩放的窗口
cv2.namedWindow("Measure Object Size", cv2.WINDOW_NORMAL)

# 计算合适的窗口大小，使其与原始比例一致
screen_res = (1920, 1080)
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)

window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

cv2.resizeWindow("Measure Object Size", window_width, window_height)

cv2.imshow("Measure Object Size", img)
cv2.setMouseCallback("Measure Object Size", click_and_measure)

cv2.waitKey(0)
cv2.destroyAllWindows()
