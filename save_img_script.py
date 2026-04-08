import numpy as np 
from PIL import Image

palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def save_image(image_array, output_image_path):
    # 创建一个新的空白图像
    colored_image = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    # 将像素值替换为相应的颜色
    for key, value in palette.items():
        colored_image[image_array == key] = value

    # 将 NumPy 数组转换回 PIL 图像
    colored_image = Image.fromarray(colored_image)

    # 保存或显示处理后的图像
    colored_image.save(output_image_path)