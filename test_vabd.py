import argparse
import subprocess

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.model import bokeh
from utils.dataset_utils import BokehDataset_test_vabd as BokehDataset_test
from utils.image_io import save_image_tensor
from utils.val_utils import AverageMeter, compute_psnr_ssim

def ssim_loss(recoverd, clean):
    assert recoverd.shape == clean.shape
    return 1 - ssim(recoverd, clean, data_range=1, size_average=True)


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


def test_Bokeh(net, dataset, task="bokeh"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=1)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch, depth_patch, mask_patch, number) in tqdm(testloader):
            degrad_patch, clean_patch, depth_patch, mask_patch, number = degrad_patch.cuda(), clean_patch.cuda(), depth_patch.cuda(), mask_patch.cuda(), number.cuda()

            restored = net(degrad_patch, depth_patch, mask_patch, number)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0, help='0 for f/1.8, 1 for f/2.8, 2 for f/8')
    parser.add_argument('--bokeh_path', type=str, default='./data/VABD/test/',
                        help='save path of test bokeh images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="", help='checkpoint save path')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)
    ckpt_path = testopt.ckpt_name
    print("CKPT name : {}".format(ckpt_path))

    net = BokehModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    print('Start testing rain streak removal...')
    bokeh_set = BokehDataset_test(testopt, train=False, num=testopt.mode)
    test_Bokeh(net, bokeh_set, task="bokeh")
