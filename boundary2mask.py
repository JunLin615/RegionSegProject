import json
import cv2
import numpy as np
import os
from PIL import Image

#边界json文件路径 eg：boundary_filename = 'dataset/add_via_region_data.json'
boundary_filename = 'E:\stage\KAUST Gan\paper work\SERS Mu\演示训练用图像\R6g\R6g.json'
#对应的图像路径 eg: images_filename = 'dataset/add'
images_filename = 'E:\stage\KAUST Gan\paper work\SERS Mu\演示训练用图像\R6g\images'
#mask保存路径 eg：output = 'dataset/add_masks'
output = 'E:\stage\KAUST Gan\paper work\SERS Mu\演示训练用图像\R6g\masks'
# 读取标注文件
with open(boundary_filename) as f:
    annotations = json.load(f)

# 创建保存掩模图像的目录
os.makedirs(output, exist_ok=True)

# 处理每一张图像
for filename, file_data in annotations.items():


    regions = file_data['regions']
    if not regions:
        continue
    # 去除文件名中的额外数字部分
    original_filename = filename.split('.png')[0] + '.png'
    print(original_filename)
    # 读取原始图像
    img = cv2.imread(images_filename+f'/{original_filename}')
    print(img)
    height, width, _ = img.shape

    # 创建空的掩模图像
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in regions.values():
        shape_attributes = region['shape_attributes']
        if shape_attributes['name'] == 'polygon':
            # 获取多边形的顶点
            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']
            points = np.array([list(zip(all_points_x, all_points_y))], dtype=np.int32)

            # 在掩模图像上绘制多边形
            cv2.fillPoly(mask, points, 255)

    # 保存掩模图像
    mask_filename = os.path.splitext(filename)[0] + '_mask.png'
    mask_path = os.path.join(output, mask_filename)
    cv2.imwrite(mask_path, mask)

    # 如果需要查看生成的掩模图像，可以使用以下代码：
    # mask_img = Image.fromarray(mask)
    # mask_img.show()
