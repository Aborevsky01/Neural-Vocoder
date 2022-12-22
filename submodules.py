import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


class LeakyConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,
                              padding=(kernel_size - 1) * dilation // 2)
        '''for p in self.conv.parameters(): 
            if p.dim() > 1: nn.init.normal_(p, 0.0, 0.02)'''

    def forward(self, x):
        return self.conv(self.leaky_relu(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations_lib):
        super().__init__()
        self.conv_relu_seq = nn.ModuleList()
        for i in range(len(dilations_lib)):
            module_pre = LeakyConv(channels, channels, kernel_size, dilations_lib[i])
            self.conv_relu_seq.append(module_pre)
            module_post = LeakyConv(channels, channels, kernel_size, 1)
            self.conv_relu_seq.append(module_post)

    def forward(self, x):
        for pre, post in zip(self.conv_relu_seq[:-1], self.conv_relu_seq[1:]):
            residual = pre(x)
            x = residual + post(residual)
        return x


class MultiResolutionFuser(nn.Module):
    def __init__(self, upsampling_coefs, up_pos, kernel_mrf_lib, dilations_mrf_lib):
        super().__init__()
        self.up_convolution = nn.ConvTranspose1d(upsampling_coefs[0] // (2 ** up_pos),
                                                 upsampling_coefs[0] // (2 ** (1 + up_pos)),
                                                 kernel_size=upsampling_coefs[up_pos + 1] * 2,
                                                 stride=upsampling_coefs[up_pos + 1],
                                                 padding=upsampling_coefs[up_pos + 1] // 2)

        self.residual_blocks = nn.ModuleList()
        for kernel_size, dilations_lib in zip(kernel_mrf_lib, dilations_mrf_lib):
            self.residual_blocks.append(
                ResidualBlock(upsampling_coefs[0] // (2 ** (1 + up_pos)), kernel_size, dilations_lib))

    def forward(self, x):
        up_x = self.up_convolution(x)
        res_x = sum([block(up_x) for block in self.residual_blocks])
        return res_x


class Generator(nn.Module):
    def __init__(self, train_config):
        super().__init__()
        self.welcome_conv = nn.Conv1d(train_config.n_mels, train_config.upsampling_coefs[0],
                                      7, 1, padding=(7 - 1) // 2)

        self.sequence = nn.ModuleList()
        for up_pos in range(len(train_config.upsampling_coefs) - 1):
            self.sequence.append(MultiResolutionFuser(train_config.upsampling_coefs,
                                                      up_pos,
                                                      train_config.kernel_global_lib,
                                                      train_config.dilations_global_lib))

        self.leaky_relu = nn.LeakyReLU()
        self.bye_conv = nn.Conv1d(train_config.upsampling_coefs[0] // (2 ** (1 + up_pos)), 1,
                                       7, 1, padding=(7 - 1) // 2)
        self.bye_activation = nn.Tanh()

    def forward(self, x):
        residual = self.welcome_conv(x)
        for mrf in self.sequence:
            residual = mrf(residual)
        out = self.bye_conv(self.leaky_relu(residual))
        return self.bye_activation(out)


class SubPeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

        self.seq = nn.ModuleList()
        prev_power = 0
        for i in range(4):
            next_power = i + 6
            self.seq.append(nn.Sequential(
                weight_norm(
                    nn.Conv2d(2 ** prev_power, 2 ** next_power, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
                    'weight'
                ),
                nn.LeakyReLU()
            ))
            prev_power = next_power

        self.bye_conv = weight_norm(nn.Conv2d(2 ** prev_power, 1024, kernel_size=(5, 1), padding=(2, 0)), 'weight')
        self.bye_activation = nn.LeakyReLU()
        self.out_conv = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)), 'weight')

    def pad_and_reshape(self, x):
        extra_timesteps = x.shape[-1] % self.period
        x_after_pad = F.pad(x, (0, self.period - extra_timesteps)) if extra_timesteps > 0 else x
        x_after_reshape = x_after_pad.view(x.shape[0], x.shape[1], x_after_pad.shape[-1] // self.period, self.period)
        return x_after_reshape

    def forward(self, x):
        local_lib = []
        x = self.pad_and_reshape(x)
        for i in range(len(self.seq)):
            x = self.seq[i](x)
            local_lib.append(x)
        x = self.bye_conv(x)
        local_lib.append(x)
        x = self.out_conv(self.bye_activation(x))
        local_lib.append(x)
        return local_lib


class PeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.period_heads = nn.ModuleList()
        for value in [2, 3, 5, 7, 11]:
            self.period_heads.append(SubPeriodDiscriminator(value))

    def forward(self, x_real, x_fake):
        library = {
            'real_out': [],
            'fake_out': [],
            'real_values': [],
            'fake_values': []
        }

        for x, name in zip([x_real, x_fake], ['real', 'fake']):
            for head in self.period_heads:
                local_lib = head(x)
                library[name + '_out'].append(local_lib)
                library[name + '_values'].append(local_lib[-1])
        return library


class SubScaleDiscriminator(nn.Module):
    def __init__(self, norm=weight_norm):
        super().__init__()
        self.seq = nn.ModuleList([
            norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=(15 - 1) // 2), 'weight'),
            nn.LeakyReLU(),
            norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=(41 - 1) // 2), 'weight'),
            nn.LeakyReLU(),
            norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=(41 - 1) // 2), 'weight'),
            nn.LeakyReLU(),
            norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=(41 - 1) // 2), 'weight'),
            nn.LeakyReLU(),
            norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=(41 - 1) // 2), 'weight'),
            nn.LeakyReLU(),
            norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=(5 - 1) // 2, ), 'weight'),
            nn.LeakyReLU(),
            norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=(3 - 1) // 2), 'weight'),
        ])

    def forward(self, x):
        local_lib = []
        for i in range(0, len(self.seq) - 1, 2):
            x = self.seq[i + 1](self.seq[i](x))
            local_lib.append(x)
        local_lib.append(torch.flatten(self.seq[-1](x), 1, -1))
        return local_lib


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_heads = nn.ModuleList(
            [SubScaleDiscriminator(spectral_norm), SubScaleDiscriminator(), SubScaleDiscriminator()])
        self.downsample = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2) for _ in range(3)])

    def forward(self, x_real, x_fake):
        library = {
            'real_out': [],
            'fake_out': [],
            'real_values': [],
            'fake_values': []
        }

        for x, name in zip([x_real, x_fake], ['real', 'fake']):
            for idx, head in enumerate(self.scale_heads):
                local_lib = head(x)
                library[name + '_out'].append(local_lib)
                library[name + '_values'].append(local_lib[-1])
                if idx < 3:
                    x = self.downsample[idx](x)
        return library
