import os
import cv2
import numpy as np
import cv2
import torch
import torch.utils.data as data

import utils

class CFP_dataset(data.Dataset):
    def __init__(self, opt):
        self.imglist, self.namelist = utils.get_lists_png_parents(opt.baseroot)

    def disentangle_label(self, label):
        # input label should be a string
        # for father and mother, the family status is 1, while children is larger than 2
        # gender: 3rd; skin color: 4th; age: 5th; emotion: 6th; glass: 7th; moustache: 8th
        # these are binary labels, and only gender, age, emotion, glass, and moustache are taken into account
        if label[:2] == '01' or label[:2] == '02':
            new_label = str(int(label[:2])) + ',' + label[2] + ',' + label[4] + ',' + label[5] + ',' + label[6] + ',' + label[7]
        else:
            new_label = '0' + ',' + label[2] + ',' + label[4] + ',' + label[5] + ',' + label[6] + ',' + label[7]
        new_label = np.fromstring(new_label, dtype = int, sep = ',')
        new_label = torch.from_numpy(new_label)
        return new_label
        
    def binarize_label(self, label):
        # index  value  meaning  value  meaning
        #   0      0     child     1     parent
        #   1      0     woman     1      man
        #   2      0     older     1    younger
        #   3      0     smile     1   not smile
        #   4      0     glass     1    no glass
        #   5      0   moustache   1  no moustache
        # Male | Eyeglasses | Mustache | Smiling: 1 = True, 0 = False
        attr = []
        if label[1] == 1:
            attr.append(1)
        if label[1] == 2:
            attr.append(0)
        if label[4] == 1:
            attr.append(1)
        if label[4] == 2:
            attr.append(0)
        if label[5] == 1:
            attr.append(1)
        if label[5] == 2:
            attr.append(0)
        if label[3] == 2:
            attr.append(1)
        if label[3] == 1 or label[3] == 3:
            attr.append(0)
        attr = np.array(attr)
        attr = torch.from_numpy(attr).float()
        return attr

    def __getitem__(self, index):
        # read string from list
        imgroot = self.imglist[index]
        imgname = self.namelist[index]
        
        # read image and get label
        img = cv2.imread(imgroot)
        imglabel = imgname.split('/')[-1][:-4]

        # normalization for images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) - 128.0) / 128.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # normalization for labels (float32)
        imglabel = self.disentangle_label(imglabel)
        imglabel = self.binarize_label(imglabel)

        return img, imglabel

    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":
    # simulate label
    label = '01122222'
    new_label = str(int(label[:2])) + ',' + label[2] + ',' + label[4] + ',' + label[5] + ',' + label[6] + ',' + label[7]
    new_label = np.fromstring(new_label, dtype = int, sep = ',')
    print('parent(1,2)/child(3,4,5,...), gender(man1/woman2), age(baby1/other23), smile(), glass(), moustache()')
    print(new_label)
    new_label = torch.from_numpy(new_label).float()
    print(new_label)
    new_label = new_label.long()
    print(new_label.dtype)

    # process label
    print('process label')
    new_label = new_label[1:]
    print(new_label)
    if new_label[4] == 2:
        new_label[4] = 0
    print(new_label)

    # simulate list
    a = [['1', '2', '3'], ['1', '2', '3'], ['1', '2', '3']]
    aa, aaa, aaaa = a[1]
    print(aa, aaa, aaaa)
