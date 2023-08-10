import platform, argparse, random
from skimage import io
import torch, os
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
parser.add_argument('--sparse_weight', type=float, default=0.1)
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
parser.add_argument('--quant', type=int, default=0)
parser.add_argument('--force_sparse', type=int, default=0)
parser.add_argument('--dim_quant', type=int, default=5000)

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
elif args.adjust_dec_std=='auto':
    dec_output_dim*=2

policy = GaussianPolicy()
encoder = Encoder(channel_dim, GRAY_SCALE, args.quant, args.ps1, args.dim_quant).to(device)
decoder = Decoder(channel_dim, orig_size=(img_size, img_size), output_dim=dec_output_dim, quant=args.quant, dim_quant=args.dim_quant).to(device)
if not args.enable_EncAgent:
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=LR)
else:
    optimizer_enc = optim.Adam(encoder.parameters(), lr=LR/10 if args.enable_EncAgent else LR)
    optimizer_dec = optim.Adam(decoder.parameters(), lr=LR)

normlize_layer = Normlize_tx(_iscomplex=_iscomplex)
channel = Channel(_iscomplex=_iscomplex)
crit = Crit(device)

train_loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
test_loader_params = {'batch_size': 50, 'shuffle': False, 'num_workers': 0}
data_path = 'D:\Cifar\cifar-10-batches-py'
dataset_train = Dataset_cifar10(data_path, GRAY_SCALE) if 'Cifar' in args.dataset else Dataset_Rand(rand_path, GRAY_SCALE)
dataset_test = Dataset_cifar10(data_path, GRAY_SCALE, mode='test')
mini_batch_loader = data.DataLoader(dataset_train,**train_loader_params)
test_batch_loader = data.DataLoader(dataset_test,**test_loader_params)

def train_SingleEncAgent(n_epi):
    global STD_EPOCH_ENC, STD_EPOCH_DEC
    encoder.train()
    decoder.train()

    os.makedirs(os.path.join(visual_path, 'ckpt_{}'.format(n_epi)), exist_ok=True)
    for idx, input_data in enumerate(mini_batch_loader):
        optimizer_enc.zero_grad()
        optimizer_dec.zero_grad()

        input_data = input_data.to(device)
        data_ = encoder(input_data)
        std_enc = None
        data_for_decoder = data_.clone().detach()
        data_n = sample_n_times(multiple_sample, data_)
        input_data_n = sample_n_times(multiple_sample, input_data)
        encoded_sample, logprobs = policy.forward_sample(data_n, std=STD_EPOCH_ENC)

        with torch.no_grad():
            data_n = normlize_layer.apply(encoded_sample)
            data_n = channel(args.ChannelType, data_n, _snr)
            data_n = decoder(data_n)
            adv, reward_sampled_trainTX = get_reward(data_n.detach().clamp(-1, 1), input_data_n)
        loss_enc = crit('tx_gaussian_sample', logprobs, adv)
        loss_enc.backward()
        data_ = normlize_layer.apply(data_for_decoder.detach())
        data_ = channel(args.ChannelType, data_, _snr)
        data_ = decoder(data_)
        std_dec = None
        if not args.enable_DecAgent:
            loss_dec = crit(args.RewardType, data_, input_data)
        else:
            data_ = sample_n_times(multiple_sample, data_)
            if std_dec is not None:
                std_dec = sample_n_times(multiple_sample, std_dec)
            input_data = sample_n_times(multiple_sample, input_data)
            decoded_sample, logprobs = policy.forward_sample(data_,std=STD_EPOCH_DEC)
            adv, reward_sampled_trainRX = get_reward(decoded_sample.detach().clamp(-1, 1), input_data, REW_TIMES)
            loss_dec = crit('tx_gaussian_sample', logprobs, adv)

        loss_dec.backward()

        optimizer_enc.step()
        optimizer_dec.step()

        if idx%10==0:
            print("epoch {a} -- loss_enc {b}  loss_dec {f} reward@enc {g} reward@dec {c}  adv {d} std_enc {e} std_dec {z}".format(a=n_epi,
                c=1-loss_dec.item() if not args.enable_DecAgent else reward_sampled_trainRX.mean().item(), e=STD_EPOCH_ENC, f=loss_dec.item(),
                  b=loss_enc.item(), d=adv.detach().cpu().numpy().__abs__().mean() if not args.Supervised_pretrain else 0,
                  g=reward_sampled_trainTX.mean().item(), z=STD_EPOCH_DEC))


def train_SingleDecAgent(n_epi):
    global STD_EPOCH_DEC
    encoder.train()
    decoder.train()

    os.makedirs(os.path.join(visual_path, 'ckpt_{}'.format(n_epi)), exist_ok=True)
    for idx, input_data in enumerate(mini_batch_loader):
        optimizer.zero_grad()
        input_data = input_data.to(device)
        data_ = encoder(input_data)
        data_quant = data_.clone()
        loss_sparse = 0
        data_ = normlize_layer.apply(data_)
        data_ = channel(args.ChannelType, data_, _snr)
        data_ = decoder(data_)
        std_ = None
        if Supervised_pretrain:
            loss = crit(args.RewardType, data_, input_data)
        else:
            data_ = sample_n_times(multiple_sample, data_)
            input_data = sample_n_times(multiple_sample, input_data)
            decoded_sample, logprobs = policy.forward_sample(data_, std=STD_EPOCH_DEC)
            adv= get_reward(decoded_sample.detach().clamp(-1, 1), input_data, REW_TIMES)
            loss = crit('tx_gaussian_sample', logprobs, adv)
        if args.force_sparse:
            loss_sparse = args.sparse_weight*torch.abs(data_quant).mean()
            loss += loss_sparse
        loss.backward()
        optimizer.step()

        if idx%10==0:
            _, reward_real = get_reward(data_.detach(), input_data, REW_TIMES)
            print("epoch {a} -- loss {b}  train total reward {c}  adv {d} std {e}  l_sparse={y}".format(a=n_epi,
                c=reward_real.mean().float(), e=STD_EPOCH_DEC,
                  b=loss.item(), d=adv.detach().cpu().numpy().__abs__().mean()/REW_TIMES if not args.Supervised_pretrain else 0,
                            y=loss_sparse.item() if args.force_sparse else 0))

def validation(test_data_loader):
    list_psnr, list_ssim = [], []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for train_imgs in enumerate(test_data_loader):
            train_imgs = train_imgs.to(device)
            output = encoder(train_imgs)
            output = normlize_layer.apply(output)
            output = channel(args.ChannelType, output, _snr)
            output = decoder(output)
            _s = ssim((train_imgs+1)/2.,(output+1)/2., data_range=1.)
            _p = psnr((train_imgs+1)/2.,(output+1)/2., data_range=1.)
            list_psnr.append(_p.cpu().numpy())
            list_ssim.append(_s.cpu().numpy())

        return sum(list_psnr)/len(list_psnr), sum(list_ssim)/len(list_ssim)

if __name__ == '__main__':
    for ep in range(args.max_epoch):
        if args.enable_EncAgent:
            train_SingleEncAgent(ep)
        else:
            train_SingleDecAgent(ep)
        _psnr, _ssim = validation(test_batch_loader)
        print('current PSNR: {}  current SSIM: {}'.format(_psnr, _ssim))
        if _psnr > psnr_max:
            print('max psnr from epoch %d  --  %.4f'%(ep, _psnr))
            print('max ssim from epoch %d  --  %.4f' % (ep, _ssim))
            psnr_max = _psnr
            torch.save(encoder.state_dict(),
                       os.path.join(save_model_path, 'encoder.pth'))
            torch.save(decoder.state_dict(),
                       os.path.join(save_model_path, 'decoder.pth'))