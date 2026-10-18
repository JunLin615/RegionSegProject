import json
import cv2
import numpy as np
import os
import argparse
import sys

def imread_safe(path):
    """支持中文路径的图片读取"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Error] 读取图片失败: {path} - {e}")
        return None

def imwrite_safe(path, img):
    """支持中文路径的图片保存"""
    try:
        cv2.imencode('.png', img)[1].tofile(path)
        return True
    except Exception as e:
        print(f"[Error] 保存图片失败: {path} - {e}")
        return False

def main():
    # 1. 配置命令行参数
    parser = argparse.ArgumentParser(description='VIA JSON 转 Mask (支持中文路径/多种形状)')
    parser.add_argument('--json', type=str, required=True, help='边界json文件路径')
    parser.add_argument('--img_dir', type=str, required=True, help='原始图像文件夹路径')
    parser.add_argument('--out_dir', type=str, required=True, help='Mask输出保存路径')

    args = parser.parse_args()

    boundary_filename = args.json
    images_filename = args.img_dir
    output = args.out_dir

    print(f"[-] 读取标注: {boundary_filename}")

    # 2. 读取标注文件
    try:
        with open(boundary_filename, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"[Error] JSON 读取失败: {e}")
        sys.exit(1)

    # 3. 智能判断 JSON 结构
    if isinstance(raw_data, dict) and '_via_img_metadata' in raw_data:
        annotations = raw_data['_via_img_metadata']
    else:
        annotations = raw_data

    # 创建输出目录
    os.makedirs(output, exist_ok=True)

    # 4. 处理循环
    # 统一转换为可迭代对象 (items)
    data_items = annotations.items() if isinstance(annotations, dict) else enumerate(annotations)

    count = 0
    for key, file_data in data_items:
        # --- 关键修正：类型过滤 ---
        if not isinstance(file_data, dict):
            continue

        # 安全获取 regions
        regions = file_data.get('regions')
        if not regions:
            continue

        # 安全获取文件名
        filename = file_data.get('filename')
        if not filename:
            filename = key # 尝试用 key 作为文件名

        if not isinstance(filename, str):
            continue

        # 构造路径 & 读取图片
        original_filename = filename.split('.png')[0] + '.png' # 去除可能的后缀干扰
        img_path = os.path.join(images_filename, original_filename)

        img = imread_safe(img_path)
        if img is None:
            # print(f"[Warning] 图片不存在或无法读取: {original_filename}")
            continue

        height, width, _ = img.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # 处理区域 (兼容 List 和 Dict 结构的 regions)
        region_list = regions.values() if isinstance(regions, dict) else regions

        for region in region_list:
            if not isinstance(region, dict): continue

            shape = region.get('shape_attributes')
            if not shape: continue

            name = shape.get('name')

            # --- 绘制逻辑 (包含多边形、圆、椭圆) ---
            if name == 'polygon':
                xs = shape.get('all_points_x', [])
                ys = shape.get('all_points_y', [])
                if len(xs) == len(ys) and len(xs) > 0:
                    points = np.array([list(zip(xs, ys))], dtype=np.int32)
                    cv2.fillPoly(mask, points, 255)

            elif name == 'circle':
                cx, cy, r = int(shape.get('cx', 0)), int(shape.get('cy', 0)), int(shape.get('r', 0))
                cv2.circle(mask, (cx, cy), r, 255, -1)

            elif name == 'ellipse':
                cx, cy = int(shape.get('cx', 0)), int(shape.get('cy', 0))
                rx, ry = int(shape.get('rx', 0)), int(shape.get('ry', 0))
                theta = shape.get('theta', 0)
                angle = np.degrees(theta)
                cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

        # 保存 Mask
        mask_filename = os.path.splitext(original_filename)[0] + '_mask.png'
        mask_path = os.path.join(output, mask_filename)

        if imwrite_safe(mask_path, mask):
            count += 1
            if count % 10 == 0:
                print(f"[-] 已处理 {count} 张...")

    print(f"--- 处理完成，共生成 {count} 张 Mask ---")

if __name__ == "__main__":
    main()
