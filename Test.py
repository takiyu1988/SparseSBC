import platform, argparse, random
import matplotlib.pyplot as plt
from skimage import io
import torch, os, pickle
import numpy as np
from model import Encoder, Decoder
from mini_batch_loader import Dataset_cifar10
import torch.optim as optim
from utils import Normlize_tx, Channel, sample_n_times, GaussianPolicy, Crit
import torch.utils.data as data
import torch.nn.functional as F
from piq import ssim, psnr
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_seeds(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

init_seeds()

def get_reward(sampled, gt, times=1):
    all_reward = 1-crit(args.RewardType, sampled, gt, reduction='none')
    all_reward = torch.tensor(all_reward.reshape([-1, multiple_sample]), device=device, requires_grad=False)
    baseline = (all_reward.sum(1, keepdim=True) - all_reward) / (all_reward.shape[1] - 1)
    advantage = all_reward - baseline
    return advantage*times, all_reward

parser = argparse.ArgumentParser()
parser.add_argument('--STD', type=float, default=0.1)
parser.add_argument('--REW_TIMES', type=float, default=10)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--Supervised_pretrain', type=int, default=1)
parser.add_argument('--enable_EncAgent', type=int, default=0)
parser.add_argument('--enable_DecAgent', type=int, default=0)
parser.add_argument('--multiple_sample', type=int, default=1)
parser.add_argument('--channel_dim', type=int, default=36)
parser.add_argument('--GRAY_SCALE', type=float, default=0)
parser.add_argument('--EXP_NAME', type=str, default='')
parser.add_argument('--dataset', type=str, default='Cifar')
parser.add_argument('--max_epoch', type=int, default=201)
parser.add_argument('--ChannelType', type=str, default='awgn')
parser.add_argument('--RewardType', type=str, default='ssim')
parser.add_argument('--quant', type=int, default=1)
parser.add_argument('--force_sparse', type=int, default=0)
parser.add_argument('--feats', type=int, default=0)
args = parser.parse_args()

REW_TIMES = args.REW_TIMES
STD = args.STD
img_size = args.img_size
GRAY_SCALE = args.GRAY_SCALE
EXP_NAME = args.EXP_NAME
channel_dim = args.channel_dim

STD_EPOCH_ENC, STD_EPOCH_DEC = STD, STD
batch_size = 64
LR = 1e-4
_snr = 10
psnr_max = -1e5
multiple_sample = args.multiple_sample
_iscomplex = True
Supervised_pretrain = args.Supervised_pretrain
if GRAY_SCALE:
    EXP_NAME = 'Supervised_GRAY_'+EXP_NAME if Supervised_pretrain else 'RL_GRAY'+EXP_NAME
else:
    EXP_NAME = 'Supervised_'+EXP_NAME if Supervised_pretrain else 'RL_'+EXP_NAME
IMAGE_DIR_PATH = ""
visual_path = './visuals/'+EXP_NAME
if not os.path.exists(visual_path): os.makedirs(visual_path)
save_model_path = './ckpts/'+EXP_NAME
if not os.path.exists(save_model_path): os.makedirs(save_model_path)
dec_output_dim=3
if GRAY_SCALE:
    dec_output_dim=1
policy = GaussianPolicy()
encoder = Encoder(channel_dim, GRAY_SCALE, args.quant).to(device)
decoder = Decoder(channel_dim, orig_size=(img_size, img_size), output_dim=dec_output_dim, quant=args.quant).to(device)
if not args.enable_EncAgent:
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=LR)
else:
    optimizer_enc = optim.Adam(encoder.parameters(), lr=LR/10 if args.enable_EncAgent else LR) # if single TXRL, use a small lr when STD_ENC is small
    optimizer_dec = optim.Adam(decoder.parameters(), lr=LR)

normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)
crit = Crit(device)

train_loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
test_loader_params = {'batch_size': 50, 'shuffle': False, 'num_workers': 0}
data_path = 'D:\Cifar\cifar-10-batches-py'
dataset_train = Dataset_cifar10(data_path, GRAY_SCALE)
dataset_test = Dataset_cifar10(data_path, GRAY_SCALE, mode='test')
test_batch_loader = data.DataLoader(dataset_test,**test_loader_params)

def validation(test_data_loader):
    os.makedirs('./pkls', exist_ok=True)
    list_psnr, list_ssim = [], []
    encoder.eval()
    decoder.eval()
    all_weights = []
    all_feats = []
    plot_flag = 0
    with torch.no_grad():
        all_bits = 0
        for batch_idx, train_imgs in enumerate(test_data_loader): # validation.dataset._data
            train_imgs = train_imgs.to(device)
            output = encoder(train_imgs)
            if args.adjust_enc_std=='auto':
                output = output[:, :output.shape[-1]//2]
            quanted = output.clone().detach()
            num_zeros = count0(quanted, reduction='mean')
            all_bits+=1-num_zeros.item()
            all_weights.append(quanted.cpu().numpy()>0.5)
            output = normlize_layer.apply(output)
            output = channel(args.ChannelType, output, _snr)
            feats = decoder.dequant(output)
            output = decoder(output)
            all_feats.append(feats.cpu().numpy())
            if args.adjust_dec_std == 'auto':
                output = output[:, :3, :]
            _s = ssim((train_imgs + 1) / 2., (output + 1) / 2., data_range=1.)
            _p = psnr((train_imgs + 1) / 2., (output + 1) / 2., data_range=1.)
            if plot_flag:
                for i, x in enumerate((output.cpu().numpy()+1)/2*255.):
                    io.imsave('./vis/test_{}.png'.format(i), x.transpose((1,2,0)).astype(np.uint8))
                for i, x in enumerate((train_imgs.cpu().numpy()+1)/2*255.):
                    io.imsave('./vis/gt_{}.png'.format(i), x.transpose((1,2,0)).astype(np.uint8))
                plot_flag = 0
            list_psnr.append(_p.cpu().numpy())
            list_ssim.append(_s.cpu().numpy())
        print('all batches: ', batch_idx+1)
        all_weights = np.vstack(all_weights)
        all_feats = np.vstack(all_feats)
        with open('./pkls/weight_{}.pkl'.format(args.EXP_NAME), 'wb') as f:
            pickle.dump(all_weights, f)
        with open('./pkls/weight_{}.pkl'.format('FeatsAtRX'), 'wb') as f:
            pickle.dump(all_feats, f)
        return sum(list_psnr)/len(list_psnr), sum(list_ssim)/len(list_ssim), all_bits/(batch_idx+1)

def count0(tens, reduction='none'):
    t = tens.clone().detach()
    a = torch.ones_like(t)
    b = torch.zeros_like(t)
    c = torch.where(t < 0.5, a, b)
    res = c.mean(-1)
    if reduction=='mean':
        res = res.mean()
    return res

if __name__ == '__main__':
    encoder.load_state_dict(torch.load(os.path.join(save_model_path, 'encoder.pth'), map_location='cpu'))
    decoder.load_state_dict(torch.load(os.path.join(save_model_path, 'decoder.pth'), map_location='cpu'))
    _psnr, _ssim, _codeweight = validation(test_batch_loader)
    print('current PSNR: {}  current SSIM: {}  current CodeWeight: {}'.format(_psnr, _ssim, _codeweight))
    validation(test_batch_loader)
