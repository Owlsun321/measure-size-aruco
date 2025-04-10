import cv2
import numpy as np
import argparse
import logging

# 自定义日志过滤器
class IgnoreSpecificLogFilter(logging.Filter):
    def __init__(self, ignore_message):
        super().__init__()
        self.ignore_message = ignore_message

    def filter(self, record):
        # 如果日志消息包含需要忽略的内容，则返回 False（不记录）
        return self.ignore_message not in record.getMessage()

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 创建日志记录器
logger = logging.getLogger()
# 添加过滤器，忽略包含 "把这行输出去掉" 的日志消息
logger.addFilter(IgnoreSpecificLogFilter("把这行输出去掉"))

class MeasurementTool:
    def __init__(self, image_path, side_length_cm):
        self.image_path = image_path
        self.side_length_cm = side_length_cm
        self.pixel_cm_ratio = None
        self.img = None
        self.points = []
        self.line_points = []
        self.dragging = False
        self.start_point = None
        self.end_point = None
        self.window_width = 1920  # 默认窗口宽度
        self.window_height = 1080  # 默认窗口高度

    def load_image_and_calibrate(self):
        """加载图像并校准像素到厘米的比率"""
        try:
            # 读取图像
            self.img = cv2.imread(self.image_path)
            if self.img is None:
                raise ValueError(f"无法读取图像，请检查路径是否正确: {self.image_path}")
            # 检测 ArUco 标记
            parameters = cv2.aruco.DetectorParameters_create()
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
            corners, _, _ = cv2.aruco.detectMarkers(self.img, aruco_dict, parameters=parameters)
            if corners:
                # 计算 ArUco 标记的周长
                aruco_perimeter = cv2.arcLength(corners[0], True)
                self.pixel_cm_ratio = (4 * self.side_length_cm) / aruco_perimeter
                logging.info(f"计算的像素到厘米比率为: {self.pixel_cm_ratio:.4f}")
                # 绘制标记边界
                int_corners = np.asarray(corners, dtype=np.int32)
                cv2.polylines(self.img, int_corners, True, (0, 0, 255), 5)
            else:
                logging.warning("未检测到 ArUco 标记，使用默认比率。")
                self.pixel_cm_ratio = 0.05
        except Exception as e:
            logging.error(f"图像加载或校准失败: {e}")
            exit(1)

    def click_and_measure(self, event, x, y, flags, param):
        """鼠标回调函数，用于绘制矩形和测量距离"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_left_button_down(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.handle_left_button_up(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.handle_right_button_down(x, y)

    def handle_left_button_down(self, x, y):
        """处理左键按下事件"""
        self.start_point = (x, y)
        self.dragging = True
        self.points.append(self.start_point)

    def handle_mouse_move(self, x, y):
        """处理鼠标移动事件"""
        self.end_point = (x, y)
        img_copy = self.img.copy()
        cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 255, 0), 2)
        cv2.imshow("Measure Object Size", img_copy)

    def handle_left_button_up(self, x, y):
        """处理左键释放事件"""
        self.dragging = False
        self.end_point = (x, y)
        if len(self.points) == 1:
            self.draw_rectangle_with_measurement(self.start_point, self.end_point)
            self.points = []

    def draw_rectangle_with_measurement(self, start_point, end_point):
        """绘制矩形并显示尺寸"""
        # 计算矩形的宽度和高度（以像素为单位）
        width_pixels = abs(end_point[0] - start_point[0])
        height_pixels = abs(end_point[1] - start_point[1])

        # 将像素转换为厘米
        width_cm = width_pixels * self.pixel_cm_ratio
        height_cm = height_pixels * self.pixel_cm_ratio

        # 输出宽高到控制台
        logging.info(f"Width: {round(width_cm, 2)} cm, Height: {round(height_cm, 2)} cm")

        # 绘制矩形
        cv2.rectangle(self.img, start_point, end_point, (0, 255, 0), 8)

        # 动态调整字体参数
        font_scale = max(0.5, min(self.window_width / 1920, 2.0))  # 根据窗口宽度动态调整
        thickness = max(1, int(font_scale * 2))  # 线条粗细随字号变化
        shadow_thickness = max(2, int(font_scale * 3))  # 阴影厚度随字号变化

        # 文本内容
        text_w = f'Width: {round(width_cm, 2)} cm'
        text_h = f'Height: {round(height_cm, 2)} cm'

        # 调整文本位置到矩形的侧面，并增加间距
        if start_point[0] < end_point[0]:  # 矩形从左到右绘制
            text_position_w = (end_point[0] + 10, start_point[1] + int(50 * self.window_width / 1920))
            text_position_h = (end_point[0] + 10, start_point[1] + int(100 * self.window_width / 1920))
        else:  # 矩形从右到左绘制
            text_position_w = (start_point[0] - int(250 * self.window_width / 1920), start_point[1] + int(50 * self.window_width / 1920))
            text_position_h = (start_point[0] - int(250 * self.window_width / 1920), start_point[1] + int(100 * self.window_width / 1920))

        # 设置字体类型
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 绘制阴影
        cv2.putText(self.img, text_w, text_position_w, font, font_scale, (0, 0, 0), shadow_thickness)
        cv2.putText(self.img, text_h, text_position_h, font, font_scale, (0, 0, 0), shadow_thickness)

        # 绘制实际文字
        cv2.putText(self.img, text_w, text_position_w, font, font_scale, (0, 255, 255), thickness)
        cv2.putText(self.img, text_h, text_position_h, font, font_scale, (0, 255, 255), thickness)

        # 更新图像显示
        cv2.imshow("Measure Object Size", self.img)

    def handle_right_button_down(self, x, y):
        """处理右键按下事件，用于测量两点之间的距离"""
        if len(self.line_points) < 2:
            self.line_points.append((x, y))
        if len(self.line_points) == 2:
            cv2.line(self.img, self.line_points[0], self.line_points[1], (255, 0, 0), 5)
            dist_pixels = np.linalg.norm(np.array(self.line_points[0]) - np.array(self.line_points[1]))
            dist_cm = dist_pixels * self.pixel_cm_ratio
            mid_point = (
                (self.line_points[0][0] + self.line_points[1][0]) // 2,
                (self.line_points[0][1] + self.line_points[1][1]) // 2,
            )
            text_d = f'Distance: {round(dist_cm, 2)} cm'

            # 输出距离到控制台
            logging.info(f"Distance: {round(dist_cm, 2)} cm")

            # 动态调整字体参数
            font_scale = max(0.5, min(self.window_width / 1920, 2.0))  # 根据窗口宽度动态调整
            thickness = max(1, int(font_scale * 2))  # 线条粗细随字号变化
            shadow_thickness = max(2, int(font_scale * 3))  # 阴影厚度随字号变化

            # 绘制阴影
            cv2.putText(self.img, text_d, mid_point, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), shadow_thickness)
            # 绘制实际文字
            cv2.putText(self.img, text_d, mid_point, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

            self.line_points = []  # 重置线段点
            cv2.imshow("Measure Object Size", self.img)

    def run(self):
        """主运行函数"""
        try:
            # 加载图像并校准
            self.load_image_and_calibrate()
            # 创建窗口
            cv2.namedWindow("Measure Object Size", cv2.WINDOW_NORMAL)
            screen_res = (1920, 1080)
            self.window_width, self.window_height = self.resize_window_to_fit_screen(screen_res)
            cv2.resizeWindow("Measure Object Size", self.window_width, self.window_height)
            # 显示图像并设置鼠标回调
            cv2.imshow("Measure Object Size", self.img)
            cv2.setMouseCallback("Measure Object Size", self.click_and_measure)
            # 等待用户操作
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # 按下 ESC 键退出
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"程序运行出错: {e}")
            exit(1)

    def resize_window_to_fit_screen(self, screen_res=(1920, 1080)):
        """调整窗口大小以适应屏幕分辨率"""
        scale_width = screen_res[0] / self.img.shape[1]
        scale_height = screen_res[1] / self.img.shape[0]
        scale = min(scale_width, scale_height)
        return int(self.img.shape[1] * scale), int(self.img.shape[0] * scale)


if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="测量工具")
    parser.add_argument("--side", type=float, required=True, help="ArUco 标记的边长（单位：厘米）")
    parser.add_argument("--image", type=str, required=True, help="图像文件的路径")
    args = parser.parse_args()

    # 初始化测量工具
    tool = MeasurementTool(image_path=args.image, side_length_cm=args.side)
    tool.run()