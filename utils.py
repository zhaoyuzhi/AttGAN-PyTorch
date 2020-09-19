import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import os

import network

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def create_generator(opt):
    if opt.pre_train:
        # Initialize the network
        generator = network.Generator(opt)
        # Init the network
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Generator is created!')
    else:
        # Initialize the network
        generator = network.Generator(opt)
        # Load a pre-trained network
        pretrained_net = torch.load(opt.finetune_path)
        load_dict(generator, pretrained_net)
        print('Generator is loaded!')
    return generator
    
def create_discriminator(opt):
    # Initialize the network
    discriminator = network.Discriminator(opt)
    # Init the network
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Discriminators is created!')
    return discriminator
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_files_png(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if '.png' in filespath and 'ori' not in filespath:
                ret.append(os.path.join(root, filespath))
    return ret

def get_files_png_parents(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if '.png' in filespath and 'ori' not in filespath:
                if filespath[:2] == '01' or filespath[:2] == '02':
                    ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def get_jpgs_png(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if '.png' in filespath and 'ori' not in filespath:
                ret.append(filespath)
    return ret

def get_jpgs_png_parents(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if '.png' in filespath and 'ori' not in filespath:
                if filespath[:2] == '01' or filespath[:2] == '02':
                    ret.append(filespath)
    return ret

def get_lists_png_parents(path):
    # read a folder, return the complete path
    ret1 = []
    ret2 = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if '.png' in filespath and 'ori' not in filespath:
                if filespath[:2] == '01' or filespath[:2] == '02':
                    ret1.append(os.path.join(root, filespath))
                    ret2.append(filespath)
    return ret1, ret2

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------------------
#            Create image pair
# ----------------------------------------
def get_basic_folder(path):
    # read a folder, return the family folder name list
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            whole_path = os.path.join(root, filespath)
            delete_len = len(whole_path.split('/')[-1]) + 1
            whole_path = whole_path[:-delete_len]
            # only save folder name (one such folder may contain many face images)
            if whole_path not in ret:
                ret.append(whole_path)
    return ret

def get_image_pairs(basic_folder_list):
    # read each folder, return each pair; ret is a 2-dimensional list and the smallest dimension represents pair
    ret = []
    for folder_name in basic_folder_list:
        # for a specific family
        for root, dirs, files in os.walk(folder_name):
            # walk this folder
            for filespath in files:
                if filespath != 'ori.png':
                    # parents
                    if int(filespath[:2]) == 1:
                        father = filespath
                    if int(filespath[:2]) == 2:
                        mother = filespath
            for filespath in files:
                if filespath != 'ori.png':
                    # children, first two integers > 2
                    if int(filespath[:2]) > 2:
                        # temp saves a training / testing pair
                        temp = []
                        temp.append(os.path.join(root, father))     # father
                        temp.append(os.path.join(root, mother))     # mother
                        temp.append(os.path.join(root, filespath))  # children
                        ret.append(temp)
    return ret

def create_image_pair_list(opt):
    basic_folder_list = get_basic_folder(opt.baseroot)
    print('basic_folder_list:')
    print(len(basic_folder_list))
    pair_list = get_image_pairs(basic_folder_list)
    print('pair_list:')
    print(len(pair_list))
    return pair_list

def save_img(father_image, mother_image, child_image, generated_image):
    # father_image
    father_image = father_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    father_image = (father_image + 1) * 128
    father_image = father_image.astype(np.uint8)[:, :, [2, 1, 0]]
    # mother_image
    mother_image = mother_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mother_image = (mother_image + 1) * 128
    mother_image = mother_image.astype(np.uint8)[:, :, [2, 1, 0]]
    # child_image
    child_image = child_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    child_image = (child_image + 1) * 128
    child_image = child_image.astype(np.uint8)[:, :, [2, 1, 0]]
    # generated_image
    generated_image = generated_image.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
    generated_image = (generated_image + 1) * 128
    generated_image = generated_image.astype(np.uint8)[:, :, [2, 1, 0]]
    # concatenate all images
    concat_image = np.concatenate((father_image, mother_image, child_image, generated_image), axis = 1)
    return concat_image

if __name__ == "__main__":
    basic_folder_list = get_basic_folder('E:\\dataset, my paper related\\Children Face Prediction dataset (CFP-Dataset)\\step 4 resized data (128)\\validation')
    print(basic_folder_list)
    print(len(basic_folder_list))
    pair_list = get_image_pairs(basic_folder_list)
    print(len(pair_list))
'''
a = torch.randn(1, 3, 4, 4)
b = torch.randn(1, 3, 4, 4)
c = (a, b)
d = repackage_hidden(c)
print(d)
'''
'''
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class GradLoss(nn.Module):
    def __init__(self, GradLoss_weight = 1):
        super(GradLoss, self).__init__()
        self.GradLoss_weight = GradLoss_weight
        self.MSEloss = nn.MSELoss()

    def forward(self, x, y):
        h_x = x.size()[2]
        w_x = x.size()[3]

        x_h_grad = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        x_w_grad = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        y_h_grad = y[:, :, 1:, :] - y[:, :, :h_x - 1, :]
        y_w_grad = y[:, :, :, 1:] - y[:, :, :, :w_x - 1]
        
        h_loss = self.MSEloss(x_h_grad, y_h_grad)
        w_loss = self.MSEloss(x_w_grad, y_w_grad)
        
        return self.GradLoss_weight * (h_loss + w_loss)
'''
