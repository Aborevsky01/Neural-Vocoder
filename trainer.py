import logging
import os
import random

import torch
from tqdm import tqdm

from logger import get_visualizer


class Trainer:
    def __init__(
            self,
            model,
            config,
            device,
            dataloaders
    ):
        super().__init__()
        self.config = config
        self.device = device

        self.model = model.to(config.device)
        self.train_dataloader = dataloaders["train"]
        self.val_loader = dataloaders['val']

        self.len_epoch = len(self.train_dataloader)
        self.start_epoch = 1
        self.epochs = config.epochs
        self.save_period = config.save_step

        self.log_step = 25

        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(logging.DEBUG)

        self.writer = get_visualizer(config, self.logger, config.visualize)
        self.checkpoint_dir = config.save_dir

    def _train_epoch(self, epoch):
        self.model.train()
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            mel_target = batch["mel"].float().squeeze().to(self.device)
            wave = batch["wave"].float().to(self.device)
            audio = self.process_batch(mel_target, wave, is_train=True)

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                for name in self.model.loss_names:
                    self.writer.add_scalar('loss_' + name, float(getattr(self.model, 'loss_' + name)))
                self.logger.debug("Train Epoch: {} {} Loss: G {:.6f} & D {:.6f}".format(
                    epoch, self._progress(batch_idx), self.model.loss_G, self.model.loss_D))
                self._log_audio(audio, 'train')

        self._evaluation_epoch(epoch)

        return

    def process_batch(self, mel_target, wave, is_train: bool):
        self.model.set_input(mel_target, wave)
        if is_train:
            self.model.optimize_parameters()
        else:
            self.model.test()
        audio = self.model.fake_A
        return audio

    def _evaluation_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.val_loader), desc='eval', total=len(self.val_loader)):
                mel_target = batch["mel"].float().squeeze().to(self.device)
                wave = batch["waves"].float().to(self.device)

                audio = self.process_batch(mel_target, wave, is_train=False)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self._log_audio(audio, 'inference')

        return

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_audio(self, audio, name):
        audio = random.choice(audio)
        self.writer.add_audio(name, audio[0], sample_rate=22050)

    def train(self):
        try:
            for epoch in range(self.start_epoch, self.epochs + 1):
                self._last_epoch = epoch
                self._train_epoch(epoch)

                if epoch % self.save_period == 0:
                    self.save_checkpoint(epoch, save_best=False, only_best=True)
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self.save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_G": self.model.optimizer_G.state_dict(),
            "optimizer_D": self.model.optimizer_D.state_dict(),
            "config": self.config,
        }
        filename = os.path.join(self.checkpoint_dir, "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            print('here')
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1

        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        self.model.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.model.optimizer_D.load_state_dict(checkpoint["optimizer_D"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
