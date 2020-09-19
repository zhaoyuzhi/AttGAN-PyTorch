import argparse
import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

import utils
import dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--load_name_ab', type = str, \
        default = './models/G_AB_LSGAN_epoch400_bs1.pth', \
            help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--load_name_ba', type = str, \
        default = './models/G_BA_LSGAN_epoch400_bs1.pth', \
            help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--save_path', type = str, default = 'results', help = 'images saving path')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type of networks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, \
        default = 'F:\\dataset, my paper related\\Children Face Prediction dataset\\step 4 resized data (128)\\validation', \
            help = 'input baseroot')
    opt = parser.parse_args()

    utils.check_path(opt.save_path)

    # Define the dataset
    image_pair_list = utils.create_image_pair_list(opt)
    testset = dataset.CFP_dataset_val(opt, image_pair_list)

    # Define the dataloader
    testloader = DataLoader(testset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    print('The overall number of images:', len(testloader))

    # Define networks
    G_AB, G_BA = utils.create_generator_val(opt)
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()

    # Forward
    for i, (father_image, mother_image, child_image, father_label, mother_label, child_label) in enumerate(testloader):
        
        # To device
        # A is for grayscale image
        # B is for color RGB image
        true_A = torch.cat((father_image, mother_image), 1)
        true_B = child_image
        true_A = true_A.cuda()
        true_B = true_B.cuda()

        # Forward
        with torch.no_grad():
            fake_B = G_AB(true_A)
            fake_A = G_BA(fake_B)
        
        # Save
        fake_B = utils.save_img(father_image, mother_image, child_image, fake_B)
        # Save
        print(i)
        imgname = os.path.join(opt.save_path, str(i) + '.png')
        cv2.imwrite(imgname, fake_B)
    