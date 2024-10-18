import argparse
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from net.model import bokeh
from utils.dataset_utils import BokehDataset_vabd as BokehDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.loss_utils import ssim_loss


class BokehModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = bokeh(use_cam=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x, x1, x2, x3):
        return self.net(x, x1, x2, x3)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name], degrad_patch, clean_patch, depth_patch, mask_patch, number) = batch
        restored = self.net(degrad_patch, depth_patch, mask_patch, number)

        loss = self.loss_fn(restored, clean_patch)
        ssim_l = ssim_loss(restored, clean_patch)
        loss = loss + ssim_l
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5, max_epochs=100)

        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--bokeh_path', type=str, default="./data/VABD/train/",
                        help='save path of test bokeh images')
    opt = parser.parse_args()
    print("Options")
    print(opt)

    trainset = BokehDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath='./ckpt2_8', every_n_epochs=1, save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=2, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=2)

    model = BokehModel()
    trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    main()

