import torch
import math
from PIL import Image
import numpy as np


# 筛选手骨骨节
def bone_filter(box):
    if box.shape[0] == 21:
        DistalPhalanx = filter(box, 6, [0, 2, 4])  # 远节指骨
        MiddlePhalanx = filter(box, 5, [0, 2])  # 中指骨
        ProximalPhalanx = filter(box, 4, [0, 2, 4])
        MCP = filter(box, 3, [0, 2])
        MCPFirst = filter(box, 2, [0])
        Ulna = filter(box, 1, [0])
        Radius = filter(box, 0, [0])
        return torch.cat([DistalPhalanx, MiddlePhalanx, ProximalPhalanx, MCP, MCPFirst, Ulna, Radius], 0)
    else:
        print("检测到的关键数据不正确！")


# box yolo侦测的框 torch.tensor
# index  要找出来的类别值
# flag  要找的指节索引列表
def filter(box, index, flag):
    box = box[torch.where(box[:, 5] == index)]
    index = box[:, 0].argsort()
    return box[index][flag]


SCORE = {'girl': {
    'Radius': [10, 15, 22, 25, 40, 59, 91, 125, 138, 178, 192, 199, 203, 210],
    'Ulna': [27, 31, 36, 50, 73, 95, 120, 157, 168, 176, 182, 189],
    'MCPFirst': [5, 7, 10, 16, 23, 28, 34, 41, 47, 53, 66],
    'MCPThird': [3, 5, 6, 9, 14, 21, 32, 40, 47, 51],
    'MCPFifth': [4, 5, 7, 10, 15, 22, 33, 43, 47, 51],
    'PIPFirst': [6, 7, 8, 11, 17, 26, 32, 38, 45, 53, 60, 67],
    'PIPThird': [3, 5, 7, 9, 15, 20, 25, 29, 35, 41, 46, 51],
    'PIPFifth': [4, 5, 7, 11, 18, 21, 25, 29, 34, 40, 45, 50],
    'MIPThird': [4, 5, 7, 10, 16, 21, 25, 29, 35, 43, 46, 51],
    'MIPFifth': [3, 5, 7, 12, 19, 23, 27, 32, 35, 39, 43, 49],
    'DIPFirst': [5, 6, 8, 10, 20, 31, 38, 44, 45, 52, 67],
    'DIPThird': [3, 5, 7, 10, 16, 24, 30, 33, 36, 39, 49],
    'DIPFifth': [5, 6, 7, 11, 18, 25, 29, 33, 35, 39, 49]
},
    'boy': {
        'Radius': [8, 11, 15, 18, 31, 46, 76, 118, 135, 171, 188, 197, 201, 209],
        'Ulna': [25, 30, 35, 43, 61, 80, 116, 157, 168, 180, 187, 194],
        'MCPFirst': [4, 5, 8, 16, 22, 26, 34, 39, 45, 52, 66],
        'MCPThird': [3, 4, 5, 8, 13, 19, 30, 38, 44, 51],
        'MCPFifth': [3, 4, 6, 9, 14, 19, 31, 41, 46, 50],
        'PIPFirst': [4, 5, 7, 11, 17, 23, 29, 36, 44, 52, 59, 66],
        'PIPThird': [3, 4, 5, 8, 14, 19, 23, 28, 34, 40, 45, 50],
        'PIPFifth': [3, 4, 6, 10, 16, 19, 24, 28, 33, 40, 44, 50],
        'MIPThird': [3, 4, 5, 9, 14, 18, 23, 28, 35, 42, 45, 50],
        'MIPFifth': [3, 4, 6, 11, 17, 21, 26, 31, 36, 40, 43, 49],
        'DIPFirst': [4, 5, 6, 9, 19, 28, 36, 43, 46, 51, 67],
        'DIPThird': [3, 4, 5, 9, 15, 23, 29, 33, 37, 40, 49],
        'DIPFifth': [3, 4, 6, 11, 17, 23, 29, 32, 36, 40, 49]
    }
}


def calcBoneAge(score, sex):
    # 根据总分计算对应的年龄
    if sex == 'boy':
        boneAge = 2.01790023656577 + (-0.0931820870747269) * score + math.pow(score, 2) * 0.00334709095418796 + \
                  math.pow(score, 3) * (-3.32988302362153E-05) + math.pow(score, 4) * (1.75712910819776E-07) + \
                  math.pow(score, 5) * (-5.59998691223273E-10) + math.pow(score, 6) * (1.1296711294933E-12) + \
                  math.pow(score, 7) * (-1.45218037113138e-15) + math.pow(score, 8) * (1.15333377080353e-18) + \
                  math.pow(score, 9) * (-5.15887481551927e-22) + math.pow(score, 10) * (9.94098428102335e-26)
        return round(boneAge, 2)
    elif sex == 'girl':
        boneAge = 5.81191794824917 + (-0.271546561737745) * score + \
                  math.pow(score, 2) * 0.00526301486340724 + math.pow(score, 3) * (-4.37797717401925E-05) + \
                  math.pow(score, 4) * (2.0858722025667E-07) + math.pow(score, 5) * (-6.21879866563429E-10) + \
                  math.pow(score, 6) * (1.19909931745368E-12) + math.pow(score, 7) * (-1.49462900826936E-15) + \
                  math.pow(score, 8) * (1.162435538672E-18) + math.pow(score, 9) * (-5.12713017846218E-22) + \
                  math.pow(score, 10) * (9.78989966891478E-26)
        return round(boneAge, 2)


# 13个关节对应的分类模型
arthrosis = {'MCPFirst': ['MCPFirst', 11],
             'MCPThird': ['MCP', 10],
             'MCPFifth': ['MCP', 10],

             'DIPFirst': ['DIPFirst', 11],
             'DIPThird': ['DIP', 11],
             'DIPFifth': ['DIP', 11],

             'PIPFirst': ['PIPFirst', 12],
             'PIPThird': ['PIP', 12],
             'PIPFifth': ['PIP', 12],

             'MIPThird': ['MIP', 12],
             'MIPFifth': ['MIP', 12],

             'Radius': ['Radius', 14],
             'Ulna': ['Ulna', 12], }


# 保留的手指顺序
CATEGORY = ['DIPFifth', 'DIPThird', 'DIPFirst', 'MIPFifth', 'MIPThird',
            'PIPFifth', 'PIPThird', 'PIPFirst', 'MCPFifth', 'MCPThird', 'MCPFirst',
            'Ulna', 'Radius']


def export(results, score, boneAge):
    report = """
    第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；
    第一近节指骨骨骺分级{}级，得{}分；第三近节指骨骨骺分级{}级，得{}分；第五近节指骨骨骺分级{}级，得{}分；
    第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；
    第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；
    尺骨分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。

    RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。""".format(
        results['MCPFirst'][0], results['MCPFirst'][1], \
        results['MCPThird'][0], results['MCPThird'][1], \
        results['MCPFifth'][0], results['MCPFifth'][1], \
        results['PIPFirst'][0], results['PIPFirst'][1], \
        results['PIPThird'][0], results['PIPThird'][1], \
        results['PIPFifth'][0], results['PIPFifth'][1], \
        results['MIPThird'][0], results['MIPThird'][1], \
        results['MIPFifth'][0], results['MIPFifth'][1], \
        results['DIPFirst'][0], results['DIPFirst'][1], \
        results['DIPThird'][0], results['DIPThird'][1], \
        results['DIPFifth'][0], results['DIPFifth'][1], \
        results['Ulna'][0], results['Ulna'][1], \
        results['Radius'][0], results['Radius'][1], \
        score, boneAge)
    return report


# 图片的等比缩放（转换成正方形）
def trans_square(image):
    img = image.convert('RGB')
    img = np.array(img, dtype=np.uint8)  # 图片转换成numpy
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img  # 如果高是长边则换轴，最后再换回来
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)  # 创建正方形背景
        background[loc:loc + short_side] = img[...]  # 数据填充在中间位置
        img = background.transpose((1, 0, 2)) if img_w < img_h else background
    return Image.fromarray(img, 'RGB')
