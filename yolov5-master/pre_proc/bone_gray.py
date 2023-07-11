import cv2
import matplotlib.pyplot as plt
import os

input_folder = r"C:\\Users\kkoo\Desktop\yolov5-master\VOC2007-RUS-CHN\JPEGImages"  # 输入文件夹路径
output_folder = r"C:\\Users\kkoo\Desktop\yolov5-master\VOC2007-RUS-CHN\JPEGImages2"  # 输出文件夹路径

file_names = os.listdir(input_folder)

for file_name in file_names:
    # 构建输入图像的完整路径
    input_image_path = os.path.join(input_folder, file_name)
    # 以灰度模式读取图像
    img1 = cv2.imread(input_image_path, 0)


    clahe = cv2.createCLAHE(tileGridSize=(5, 5))
    dst1 = clahe.apply(img1)


    hist1 = cv2.calcHist([dst1], [0], None, [256], [0, 256])
    plt.plot(hist1, color="b")

    # 构建输出图像的完整路径
    output_image_path = os.path.join(output_folder, file_name)
    # 保存处理后的图像
    cv2.imwrite(output_image_path, dst1)