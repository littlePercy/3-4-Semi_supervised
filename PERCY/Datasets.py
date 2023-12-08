import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms
import random
import numpy as np


def get_dataPath(root, isTraining):
    # 返回图像路径列表
    if isTraining:
        root1 = os.path.join(root + "/source_imgs")
        imgPath1 = list(map(lambda x: os.path.join(root1, x), os.listdir(root1)))
        # imgPath1.sort()
        root2 = os.path.join(root + "/target_imgs")
        imgPath2 = list(map(lambda x: os.path.join(root2, x), os.listdir(root2)))
        # imgPath2.sort()

    return imgPath1, imgPath2, len(imgPath2)


class Sem_segDataset(data.Dataset):

    def __init__(self, root, isTraining=True):
        self.root = root
        self.imgList1, self.imgList2, self.imgLength2 = get_dataPath(root, isTraining)
        self.isTraining = isTraining

    def __getitem__(self, index):
        imgPath1 = self.imgList1[index]
        filename1 = imgPath1.split('/')[-1]
        imagePath2 = self.imgList2[index % self.imgLength2]
        filename2 = imagePath2.split('/')[-1]
        img_s = Image.open(imgPath1)
        img_s = img_s.convert('RGB')
        img_t = Image.open(imagePath2)
        img_t = img_t.convert('RGB')
        simple_transform = transforms.ToTensor()
        if self.isTraining:
            segPath1 = self.root + '/source_masks/' + filename1
            seg_s = Image.open(segPath1)
#             segPath2 = self.root + '/Optovue/train/seg_label/' + filename2
#             seg_t = Image.open(segPath2)
            """
            Data augmentation for training date, if needed.
            """
            p1 = random.randint(-10, 10)
            p2 = random.randint(0, 1)
            flip_transform = transforms.RandomVerticalFlip(p2)
            img_s = img_s.rotate(p1)
            seg_s = seg_s.rotate(p1)
            img_t = img_t.rotate(p1)
            seed = np.random.randint(2147483647)
            random.seed(seed)
            img_s = flip_transform(img_s)
            seg_s = flip_transform(seg_s)
            img_t = flip_transform(img_t)
#         else:
#             depthPath1 = self.root + '/Zeiss/test/depth_label/' + filename1
#             segPath1 = self.root + '/Zeiss/test/seg_label/' + filename1
#             segPath2 = self.root + '/Optovue/test/seg_label/' + filename2
#             depth_s = Image.open(depthPath1)
#             seg_s = Image.open(segPath1)
#             seg_t = Image.open(segPath2)
        img_s = simple_transform(img_s)
        seg_s = simple_transform(seg_s)
        img_t = simple_transform(img_t)

        return img_s, seg_s, img_t

    def __len__(self):
        """
        返回总的图像数量
        """
        return len(self.imgList1)# 返回源域数量
