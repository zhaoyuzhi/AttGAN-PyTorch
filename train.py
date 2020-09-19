import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train ot not')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 50, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--finetune_path', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 200, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0001, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 50, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 200000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_recon', type = float, default = 100, help = 'coefficient for Reconstruction Loss')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'coefficient for GAN Loss')
    parser.add_argument('--lambda_class', type = float, default = 10, help = 'coefficient for Classification Loss')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--norm_g', type = str, default = 'bn', help = 'normalization type of networks')
    parser.add_argument('--norm_d', type = str, default = 'in', help = 'normalization type of networks')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = 'in channels for the main stream of generator')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channels for the main stream of generator')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--attr_channels', type = int, default = 4, help = 'noise channels')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, \
        default = 'F:\\dataset, my paper related\\Children Face Prediction dataset (CFP-Dataset)\\step 4 resized data\\train', \
            help = 'input baseroot')
    opt = parser.parse_args()
    print(opt)

    '''
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    '''
    
    # ----------------------------------------
    #                 CycleGAN
    # ----------------------------------------
    trainer.CycleGAN_LSGAN(opt)
