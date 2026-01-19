import cv2

# 读取PNG图像
image_path = 'dataset/a001,R6G,-8,1,8,8_combined.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 检查图像是否成功加载
if image is None:
    print("Error: Image not loaded.")
else:
    # OpenCV读取的图像默认是BGR格式，不需要转换
    # 如果图像是RGB格式，可以使用以下代码进行转换
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 保存转换后的图像
    output_path = 'dataset/converted_image.png'
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")
