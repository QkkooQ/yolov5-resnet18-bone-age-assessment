import os
import random
from PIL import Image

# 生成train.txt 和 val.txt
import cv2


def save_file(list, path, name):
    myfile = os.path.join(path, name)
    if os.path.exists(myfile):
        os.remove(myfile)
    with open(myfile, "w") as f:
        f.writelines(list)

#数据预处理（去雾）
def opt_img(img_path):
    img = cv2.imread(img_path,0)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(3,3))
    img = clahe.apply(img)
    cv2.imwrite(img_path,img)

#数据增样
def img_rotate(img_path,flag=5):
    img = Image.open(img_path)
    for i in range(flag):
        rota = random.randint(-45,45)
        dst = img.rotate(rota)
        file_path_name,_ = img_path.split(".")
        dst.save(file_path_name+f"{i}.png")

pic_path_folder = r"D:\dataset\arthrosis"

print("开始图像预处理")
for pic_folder in os.listdir(pic_path_folder):
    data_path = r"D:/dataset/arthrosis/" + pic_folder
    num_class = len(os.listdir(data_path))
    for folder in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, folder)):
            continue
        img_lists = os.listdir(os.path.join(data_path, folder))
        for index, img in enumerate(img_lists):
            #去雾
            opt_img(os.path.join(data_path, folder, img))
            #增样
            img_rotate(os.path.join(data_path, folder, img))
print("图像预处理完成！")
print("开始切分数据集！")
for pic_folder in os.listdir(pic_path_folder):
    data_path = r"D:/dataset/arthrosis/" + pic_folder

    num_class = len(os.listdir(data_path))
    train_list = []
    val_list = []
    train_ratio = 0.9
    for folder in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, folder)):
            continue
        train_nums = len(os.listdir(os.path.join(data_path, folder))) * train_ratio
        img_lists = os.listdir(os.path.join(data_path, folder))
        random.shuffle(img_lists)
        for index, img in enumerate(img_lists):
            if index < train_nums:
                train_list.append(os.path.join(data_path, folder, img) + ' ' + str(int(folder) - 1) + '\n')
            else:
                val_list.append(os.path.join(data_path, folder, img) + ' ' + str(int(folder) - 1) + '\n')

    random.shuffle(train_list)
    random.shuffle(val_list)
    save_file(train_list, data_path, 'train.txt')
    save_file(val_list, data_path, 'val.txt')
print("完成")