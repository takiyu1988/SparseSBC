import torch
from torch.distributions import Normal
from piq import ssim

class Normlize_tx:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex
    def apply(self, _input):
        _dim = _input.shape[1]//2 if self._iscomplex else _input.shape[1]
        _norm = _dim**0.5 / torch.sqrt(torch.sum(_input**2, dim=1))
        return _input*_norm.view(-1,1)

class Channel:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex

    def __call__(self, mode, *args):
        return getattr(self, mode)(*args)

    def ideal_channel(self, _input, _snr):
        return _input

    def awgn(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5
        _input = _input + torch.randn_like(_input) * _std
        return _input

    def phase_invariant_fading(self, _input, _snr):
        _std = (10**(-_snr/10.)/2)**0.5 if self._iscomplex else (10**(-_snr/10.))**0.5
        if self._iscomplex:
            _mul = (torch.randn(_input.shape[0], 1)**2/2. + torch.randn(_input.shape[0], 1)**2/2.)**0.5
        else:
            _mul = (torch.randn(_input.shape[0], 1)**2 + torch.randn(_input.shape[0], 1)**2) ** 0.5
        _input = _input * _mul.to(_input)
        _input = _input +  torch.randn_like(_input) * _std
        return _input

class GaussianPolicy():

    def forward(self, x, std=0.1):
        return x + torch.randn_like(x) * std

    def forward_sample(self, mean, std=0.1):
        dist = Normal(mean, std)
        action = dist.sample()
        ln_prob = dist.log_prob(action)
        return action, ln_prob

class Crit:

    def __init__(self, device):
        self.device = device

    def __call__(self, mode, *args, **kwargs):
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _mse(self, x, y, reduction='mean'):
        res = torch.square(x-y).mean(list(range(x.dim()))[1:])
        if reduction=='mean':
            res = res.mean()
        return res

    def _l1(self, x, y, reduction='mean'):
        res = torch.abs(x-y).mean(list(range(x.dim()))[1:])/2
        if reduction=='mean':
            res = res.mean()
        return res

    def _ssim(self, x, y, reduction='mean'):
        return 1 - ssim((x + 1) / 2, (y + 1) / 2, reduction=reduction, data_range=1.)

    def _rl(self, seq_logprobs, seq_masks, reward):
        output = - seq_logprobs * seq_masks * reward
        output = torch.sum(output) / torch.sum(seq_masks)
        return output

    def _tx_gaussian_sample(self, log_samples, reward):
        reward = reward.view(log_samples.shape[0], -1)
        return -(reward*log_samples.view(reward.shape[0], -1)).mean()

def sample_n_times(n, x):
        if n>1:
            x = x.unsqueeze(1) # Bx1x...
            x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
            x = x.reshape(x.shape[0]*n, *x.shape[2:])
        return x
