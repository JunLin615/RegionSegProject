import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import sys
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_model(model_path, device):
    model = deeplabv3_resnet50(pretrained=False)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model

def load_images(image_dir, transform):
    images = []
    image_paths = []
    for img_name in os.listdir(image_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            images.append(image)
            image_paths.append(img_path)
    return images, image_paths

def process_and_save_results(model, images, image_paths, save_dir, device):
    mask_save_dir = os.path.join(save_dir, 'masks')
    combined_save_dir = os.path.join(save_dir, 'combined')

    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(combined_save_dir, exist_ok=True)

    # 设置一个最小面积阈值，过滤掉太小的噪点区域（像素数）
    min_area_threshold = 50

    with torch.no_grad():
        for image, img_path in tqdm(zip(images, image_paths), total=len(images), desc="Processing images",
                                    file=sys.stdout):
            image_tensor = image.to(device).unsqueeze(0)
            output = model(image_tensor)['out']
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = (pred > 0.5).astype(np.uint8)

            # 保存预测结果 (Mask)
            mask_save_path = os.path.join(mask_save_dir, os.path.basename(img_path).replace('.png', '_pred.png'))
            plt.imsave(mask_save_path, pred, cmap='gray')

            # 读取原始图像
            if not os.path.exists(img_path):
                print(f"Image {img_path} not found, skipping.")
                continue
            original_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if original_image is None:
                print(f"Failed to load image {img_path}, skipping.")
                continue
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # 调整pred的大小以匹配original_image
            pred_resized = cv2.resize(pred, (original_image.shape[1], original_image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

            # 创建彩色mask
            color_mask = np.zeros_like(original_image)
            color_mask[pred_resized == 1] = [255, 0, 0]  # 红色标注

            # 合成原图和mask
            combined_image = cv2.addWeighted(original_image, 0.7, color_mask, 0.3, 0)

            # --- 修改开始：多区域重心检测 ---

            # 1. 查找轮廓 (RETR_EXTERNAL 只检测外轮廓，忽略孔洞)
            contours, _ = cv2.findContours(pred_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 2. 遍历每个轮廓计算重心
            for contour in contours:
                # 过滤掉面积过小的噪点
                if cv2.contourArea(contour) < min_area_threshold:
                    continue

                # 计算当前轮廓的矩
                moments = cv2.moments(contour)

                if moments["m00"] != 0:
                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])

                    # 绘制中心点
                    cv2.circle(combined_image, (cX, cY), 5, (255, 255, 255), -1)
                    # 绘制坐标文字
                    cv2.putText(combined_image, f"({cX}, {cY})", (cX, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # --- 修改结束 ---

            # 保存合成结果
            combined_save_path = os.path.join(combined_save_dir,
                                              os.path.basename(img_path).replace('.png', '_combined.png'))
            combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

            cv2.imencode('.png', combined_image)[1].tofile(combined_save_path)

def main(model_dir, data_dir, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = load_model(model_path, device)
    # print(model) # 注释掉以保持控制台整洁，需要看模型结构可取消注释

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((640, 1024)),
        transforms.ToTensor(),
    ])

    # 加载待处理数据
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    images, image_paths = load_images(data_dir, transform)

    if not images:
        print("No images found to process.")
        return

    # 处理数据并保存结果
    process_and_save_results(model, images, image_paths, save_dir, device)
    print("Processing complete.")

if __name__ == "__main__":
    # 使用原始路径 (请根据实际情况确认路径是否存在)
    model_dir = r'E:\stage\KAUST Gan\paper work\SERS Mu\RegionIdentification\R6G251231\model_save\20251231_181042'
    data_dir = r'E:\stage\KAUST Gan\paper work\SERS Mu\RegionIdentification\R6G251231\images'
    save_dir = r'E:\stage\KAUST Gan\paper work\SERS Mu\RegionIdentification\R6G251231\reg'

    main(model_dir, data_dir, save_dir)
