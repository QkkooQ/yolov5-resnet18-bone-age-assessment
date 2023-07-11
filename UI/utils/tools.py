from PIL import Image
import numpy as np

def trans_square(image):
    r"""Open the image using PIL."""
    img = image.convert('RGB')
    img = np.array(img, dtype=np.uint8)  # 图片转numpy
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img  # 如果高是长边则换轴，最后再换回来
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)  # 创建正方形背景
        background[loc: loc + short_side] = img[...]  # 数据填充在中间位置
        img = background.transpose((1, 0, 2)) if img_w < img_h else background
    return Image.fromarray(img, 'RGB')
