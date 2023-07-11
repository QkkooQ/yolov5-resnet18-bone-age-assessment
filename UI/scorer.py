import torch
import common
from torchvision import models
from torch import nn
import cv2
from bone_filter_utils import bone_filter
import os
from PIL import Image
from torchvision.transforms import Compose,ToTensor,Resize,InterpolationMode

data_transforms = Compose([Resize(size=(224,224),interpolation=InterpolationMode.NEAREST),ToTensor()])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    yolov5_model = torch.hub.load(r"yolov5-master", "custom",
                                  r"yolov5-master\runs\train\best.pt", source="local")
    yolov5_model.conf = 0.7
    yolov5_model.eval = ()

    cls_models={}
    for i, name in enumerate(common.CATEGORY):
        model_name = common.arthrosis[name][0]
        if model_name in cls_models:
            continue

        net = models.resnet18()
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(in_features=512, out_features=common.arthrosis[name][1], bias=True)
        net.load_state_dict(torch.load("params/{}_best.pth".format(model_name), map_location=DEVICE))
        net.eval()
        cls_models[model_name] = net
    print("model loaded!!")
    return yolov5_model, cls_models

def detect_img(yolov5_model,cls_models,img_path,sex):
    print("detect started!!")
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(tileGridSize=(5,5))
    dst1 = clahe.apply(gray)
    results = yolov5_model(dst1)

    out = results.xyxy[0]
    out = bone_filter(out)
    out = out.cpu().numpy()
    print(out)


    score_all = 0
    bone_results={}
    for i,name in enumerate(common.CATEGORY):
        x1 = int(out[i][0])
        y1 = int(out[i][1])
        x2 = int(out[i][2])
        y2 = int(out[i][3])
        print(x1,y1,x2,y2)
        img_roi = img[y1:y2,x1:x2]
        if not os.path.exists("cutpictures"):
            os.makedirs("cutpictures")
        save_path = "./cutpictures/{}.png".format(name)
        cv2.imwrite(save_path,img_roi)


        im = Image.open(save_path)
        im = common.trans_square(im)
        im = im.convert("L")
        im = data_transforms(im)


        im = torch.unsqueeze(im, dim=0)
        cls_net = cls_models[common.arthrosis[name][0]]
        cls_out = cls_net(im)
        bone_index = int(cls_out.argmax(dim=1))
        print(name + "'s level is: " + str(bone_index + 1))

        score = common.SCORE[sex][name][bone_index]
        bone_results[name] = [bone_index + 1, int(score)]
        score_all += score

    bone_age = common.calcBoneAge(score_all, sex)
    export = common.export(bone_results, score_all, bone_age)
    return export
