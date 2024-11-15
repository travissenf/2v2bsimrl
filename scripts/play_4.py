import pygame
import sys
import cv2
import numpy as np

def draw_pacman(screen, x, y, z):
    """
    在屏幕上绘制 pacman_blue.png 及其阴影，同时显示背景。
    :param screen: Pygame 显示窗口
    :param x, y, z: 角色在 3D 空间中的位置
    """
    # 透视变换矩阵
    M = np.array([[4.44444444e-01, -4.17333333e-01, 2.60833333e+02],
                  [1.51340306e-17, -1.11111111e-01, 3.47222222e+02],
                  [-0.00000000e+00, -8.88888889e-04, 1.00000000e+00]])

    # 绘制背景
    screen.blit(background_surface, (0, 0))

    # 使用透视变换矩阵转换角色位置
    y_shadow = y  # 简单模拟 z 影响 y 坐标
    # y -= z
    y += 125  # 调整 y 偏移量
    y_shadow += 125

    point = np.array([[x, y]], dtype='float32')
    point = np.array([point])

    point_shadow = np.array([[x, y_shadow]], dtype='float32')
    point_shadow = np.array([point_shadow])

    transformed_point = cv2.perspectiveTransform(point, M)
    transformed_point_shadow = cv2.perspectiveTransform(point_shadow, M)

    x_t, y_t = transformed_point[0][0]
    y_t -= z

    x_t_shadow, y_t_shadow = transformed_point_shadow[0][0]

    # 设置初始的 Pacman 半径
    PLAYER_CIRCLE_SIZE = 15  # 半径为 15 像素

    # 计算 Pacman 图片的缩放比例
    scale_ratio = 1  # 防止比例过小
    new_width = int(pacman_original_width * scale_ratio)
    new_height = int(pacman_original_height * scale_ratio)

    # 确保新的尺寸至少为 5 像素，防止崩溃
    new_width = max(new_width, 5)
    new_height = max(new_height, 5)

    # 缩放 Pacman 图片
    scaled_pacman = pygame.transform.smoothscale(pacman_img, (new_width, new_height))

    # 调整位置以使图片中心与 (x_t, y_t) 对齐
    pacman_pos = (int(x_t - new_width / 2), int(y_t - new_height / 2))

    # 绘制阴影
    shadow_alpha = max(int(255 * (1 - (z - z1) / (z2 - z1))), 50)
    shadow_width = new_width
    shadow_height = max(int(new_height / 10), 3)

    shadow_surface = pygame.Surface((shadow_width, shadow_height), pygame.SRCALPHA)
    shadow_surface.set_alpha(shadow_alpha)
    pygame.draw.ellipse(shadow_surface, (0, 0, 0), (0, 0, shadow_width, shadow_height))
    shadow_pos = (int(x_t_shadow - shadow_width / 2), int(y_t_shadow + new_height / 2))
    screen.blit(shadow_surface, shadow_pos)

    # 绘制 Pacman 图片
    screen.blit(scaled_pacman, pacman_pos)

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
pygame.display.set_caption("Moving Pacman on Background")

# 设置初始的 Pacman 半径
PLAYER_CIRCLE_SIZE = 15  # 半径为 15 像素

# 加载 Pacman 图片并调整为指定的初始大小
pacman_img = pygame.image.load('Pacman_blue.png').convert_alpha()
pacman_img = pygame.transform.smoothscale(pacman_img, (PLAYER_CIRCLE_SIZE * 2, PLAYER_CIRCLE_SIZE * 2))
pacman_original_width, pacman_original_height = pacman_img.get_size()

# 设置起点和终点坐标（原始图像坐标系）
x1, y1, z1 = 0, 0, 10
x2, y2, z2 = 500, 500, 50

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

    # 绘制 Pacman 和背景
    draw_pacman(screen, x, y, z)

    # 更新显示
    pygame.display.flip()
    clock.tick(30)  # 每秒 30 帧

# 退出 Pygame
pygame.quit()
sys.exit()
