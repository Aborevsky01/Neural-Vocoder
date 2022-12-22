from torch.nn.utils import clip_grad_norm_

from loss import *
from submodules import *


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class HifiGAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.device
        self.save_dir = config.save_dir
        self.config = config

        self.loss_names = ['G', 'D', 'generator', 'feature', 'mel']
        self.visual_names = ['real_A', 'fake_A', 'real_B']
        self.model_names = ['G', 'D']
        self.wav2mel = config.wav2mel.to(self.device)

        self.netG = Generator(config)

        self.msd = ScaleDiscriminator()
        self.mpd = PeriodDiscriminator()

        self.criterionL1 = torch.nn.L1Loss()
        self.criterion_feature = FeatureLoss()
        self.criterion_D = GanLoss('D')
        self.criterion_G = GanLoss('G')

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config.lr,
                                            betas=(0.8, 0.999), weight_decay=config.weight_decay)
        self.optimizer_D = torch.optim.Adam(list(self.msd.parameters()) + list(self.mpd.parameters()),
                                            lr=config.lr, betas=(0.8, 0.999), weight_decay=config.weight_decay)

        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=.9)
        self.scheduler_D = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D, gamma=.9)

    def set_input(self, mel, wave):
        self.real_A = wave.to(self.device)
        self.real_B = mel.to(self.device)

    def forward(self):
        self.fake_A = self.netG(self.real_B)
        self.real_A = F.pad(self.real_A, (0, self.fake_A.shape[2] - self.real_A.shape[2]))

    def test(self):
        with torch.no_grad():
            self.forward()

    def backward_D(self):
        library_period = self.mpd(self.real_A, self.fake_A.detach())
        library_scale = self.msd(self.real_A, self.fake_A.detach())

        self.loss_D = self.criterion_D([library_scale['real_values'], library_scale['fake_values']]) + \
                      self.criterion_D([library_period['real_values'], library_period['fake_values']])
        self.loss_D.backward()

    def backward_G(self):

        fake_mel = self.wav2mel(self.fake_A).squeeze()
        real_B = F.pad(self.real_B, (0, fake_mel.shape[-1] - self.real_B.shape[-1]))
        self.loss_mel = self.criterionL1(fake_mel, real_B)

        library_period = self.mpd(self.real_A, self.fake_A)
        library_scale = self.msd(self.real_A, self.fake_A)

        self.loss_feature = self.criterion_feature(library_period['real_out'], library_period['fake_out']) + \
                            self.criterion_feature(library_scale['real_out'], library_scale['fake_out'])

        self.loss_generator = self.criterion_G(library_period['fake_values']) + \
                              self.criterion_G(library_scale['fake_values'])

        self.loss_G = self.loss_generator + 2 * self.loss_feature + self.loss_mel * 45
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        set_requires_grad(self.msd, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        # self._clip_grad_norm()
        self.optimizer_D.step()

        set_requires_grad(self.msd, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        # self._clip_grad_norm()
        self.optimizer_G.step()

    def _clip_grad_norm(self):
        if hasattr(self.config, 'grad_norm_clip'):
            clip_grad_norm_(
                self.parameters(), self.config.grad_norm_clip
            )
