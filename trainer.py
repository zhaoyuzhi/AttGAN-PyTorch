import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils

def CycleGAN_LSGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_BCE = torch.nn.BCEWithLogitsLoss().cuda()

    # Initialize networks
    G = utils.create_generator(opt)
    D = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        G = nn.DataParallel(G)
        G = G.cuda()
        D = nn.DataParallel(D)
        D = D.cuda()
    else:
        G = G.cuda()
        D = D.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(G.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(D.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, G, D):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(G.module, 'AttnGAN_parent_G_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                    torch.save(D.module, 'AttnGAN_parent_D_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(G.module, 'AttnGAN_parent_G_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                    torch.save(D.module, 'AttnGAN_parent_D_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(G, 'AttnGAN_parent_G_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                    torch.save(D, 'AttnGAN_parent_D_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(G, 'AttnGAN_parent_G_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                    torch.save(D, 'AttnGAN_parent_D_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                    print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.CFP_dataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (img, imglabel) in enumerate(dataloader):

            # To device
            img = img.cuda()
            idx = torch.randperm(len(imglabel))
            imglabel_fake = imglabel[idx].contiguous()
            imglabel = imglabel.cuda()
            imglabel_fake = imglabel_fake.cuda()

            # ------------------------------- Train Generator -------------------------------
            optimizer_G.zero_grad()

            # Forward
            img_recon, img_fake = G(img, imglabel, imglabel_fake)
            out_adv, out_class = D(img_fake)

            # Recon Loss
            loss_recon = criterion_L1(img_recon, img)
            
            # WGAN loss
            loss_gan = - torch.mean(out_adv)

            # Classification Loss
            loss_class = criterion_BCE(out_class, imglabel_fake)

            # Overall Loss and optimize
            loss = opt.lambda_recon * loss_recon + opt.lambda_gan * loss_gan + opt.lambda_class * loss_class
            loss.backward()
            optimizer_G.step()

            # ------------------------------- Train Discriminator -------------------------------
            optimizer_D.zero_grad()

            # Forward
            img_recon, img_fake = G(img, imglabel, imglabel_fake)
            out_adv_fake, out_class_fake = D(img_fake.detach())
            out_adv_true, out_class_true = D(img.detach())

            # WGAN loss
            loss_gan = torch.mean(out_adv_fake) - torch.mean(out_adv_true)

            # Classification Loss
            loss_class = criterion_BCE(out_class_true, imglabel)

            # Overall Loss and optimize
            loss = loss_gan + loss_class
            loss.backward()
            optimizer_D.step()
            
            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Recon Loss: %.4f] [GAN Loss: %.4f] [Class Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_recon.item(), loss_gan.item(), loss_class.item(), time_left))
            
            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), G, D)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)

if __name__ == "__main__":

    att_a = torch.randn(5)
    print(att_a)
    
    idx = torch.randperm(len(att_a))
    att_b = att_a[idx].contiguous()
    print(att_b)
    