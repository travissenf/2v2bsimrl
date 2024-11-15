import pygame
import sys
import cv2
import numpy as np

def draw_dot(screen, x, y, z):
    """
    在屏幕上绘制红色圆点及其阴影，同时显示背景。
    :param screen: Pygame 显示窗口
    :param x, y, z: 红点在 3D 空间中的位置
    """
    # 透视变换矩阵
    M = np.array([[4.44444444e-01, -4.17333333e-01, 2.60833333e+02],
                  [1.51340306e-17, -1.11111111e-01, 3.47222222e+02],
                  [-0.00000000e+00, -8.88888889e-04, 1.00000000e+00]])

    # 绘制背景
    screen.blit(background_surface, (0, 0))

    # 使用透视变换矩阵转换红点位置
    x += z  # 简单模拟 z 影响 x 坐标
    y += 125  # 调整 y 偏移量
    point = np.array([[x, y]], dtype='float32')
    point = np.array([point])
    transformed_point = cv2.perspectiveTransform(point, M)
    x_t, y_t = transformed_point[0][0]

    # 设置最小的红点半径
    radius = max(int(10 * (1 / z)), 5)  # 确保红点半径至少为 5 像素

    # 绘制阴影
    shadow_alpha = max(int(255 * (1 - (z - z1) / (z2 - z1))), 50)
    shadow_radius_x = max(int(z / 4), 5)
    shadow_radius_y = max(int(z / 4), 3)

    shadow_surface = pygame.Surface((shadow_radius_x * 2, shadow_radius_y * 2), pygame.SRCALPHA)
    shadow_surface.set_alpha(shadow_alpha)
    pygame.draw.ellipse(shadow_surface, (0, 0, 0), (0, 0, shadow_radius_x * 2, shadow_radius_y * 2))
    shadow_pos = (int(x_t - shadow_radius_x), int(y_t + shadow_radius_y))
    screen.blit(shadow_surface, shadow_pos)

    # 绘制红色圆点
    pygame.draw.circle(screen, (255, 0, 0), (int(x_t), int(y_t)), radius)


# 初始化 Pygame
pygame.init()

# 加载背景图片
background_img = cv2.imread("warped_court.jpg")
if background_img is None:
    print("无法找到 warped_court.jpg 文件，请确保文件在正确的路径。")
    pygame.quit()
    sys.exit()

# 将 OpenCV 图像转换为 Pygame Surface
background_surface = pygame.surfarray.make_surface(
    np.transpose(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB), (1, 0, 2))
)

# 获取背景的尺寸
background_height, background_width = background_img.shape[:2]

# 设置屏幕尺寸
screen = pygame.display.set_mode((background_width, background_height))
pygame.display.set_caption("Moving Red Dot on Background")

# 设置起点和终点坐标（原始图像坐标系）
x1, y1, z1 = 0, 0, 10
x2, y2, z2 = 939, 500, 100

# 初始化位置和大小
x, y, z = x1, y1, z1
steps = 100  # 将移动分为 100 步
dx = (x2 - x1) / steps
dy = (y2 - y1) / steps
dz = (z2 - z1) / steps
step_count = 0

# 主循环
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 计算新位置
    if step_count <= steps:
        x += dx
        y += dy
        z += dz
        step_count += 1

    # 绘制点和背景
    draw_dot(screen, x, y, z)

    # 更新显示
    pygame.display.flip()
    clock.tick(30)  # 每秒 30 帧

# 退出 Pygame
pygame.quit()
sys.exit()