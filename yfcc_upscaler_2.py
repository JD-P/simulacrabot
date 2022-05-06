#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import sys

from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange
import wandb

sys.path.append('./ResizeRight')

import resize_right


# Define utility functions

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


# Define the diffusion noise schedule

def get_alphas_sigmas(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


# Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        c = 192  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(   # 1x
            ResConvBlock(3 + 3 + 16, cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,  # 2x
                ResConvBlock(cs[0], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                SkipBlock([
                    self.down,  # 4x
                    ResConvBlock(cs[1], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    SkipBlock([
                        self.down,  # 8x
                        ResConvBlock(cs[2], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            self.down,  # 16x
                            ResConvBlock(cs[3], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[3]),
                            self.up,
                        ]),
                        ResConvBlock(cs[3] * 2, cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[2]),
                        self.up,
                    ]),
                    ResConvBlock(cs[2] * 2, cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[1]),
                    self.up,
                ]),
                ResConvBlock(cs[1] * 2, cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[0]),
                self.up,
            ]),
            ResConvBlock(cs[0] * 2, cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], 3, is_last=True),
        )

    def forward(self, input, t, low_res):
        low_res_big = F.interpolate(low_res, input.shape[2:], mode='bilinear', align_corners=False)
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        return self.net(torch.cat([input, low_res_big, timestep_embed], dim=1))


@torch.no_grad()
def sample(model, x, steps, eta, extra_args={}):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i], **extra_args).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred


class ImageDataset(data.Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform if transform is not None else nn.Identity()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            try:
                return (self.transform(Image.open(self.paths[index])),)
            except (OSError, ValueError,
                    Image.DecompressionBombError, Image.UnidentifiedImageError) as err:
                print(f'Bad image, skipping: {index} {self.paths[index]} {err!s}', file=sys.stderr)
                return (self[(index + 1) % len(self)],)
        except Exception as err:
            print(f'{type(err).__name__}: {err}', file=sys.stderr)
            raise


class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ResizeIfLarger:
    def __init__(self, size, mode):
        self.size = size
        self.mode = mode

    def __call__(self, image):
        if min(image.size) > self.size:
            return TF.resize(image, self.size, self.mode)
        return image


class ResizeIfSmaller:
    def __init__(self, size, mode):
        self.size = size
        self.mode = mode

    def __call__(self, image):
        if min(image.size) < self.size:
            return TF.resize(image, self.size, self.mode)
        return image


class RandomResizeIfLarger:
    def __init__(self, min_size, max_size, mode):
        self.min_size = min_size
        self.max_size = max_size
        self.mode = mode

    def __call__(self, image):
        size = torch.randint(self.min_size, self.max_size + 1, ()).item()
        if min(image.size) > size:
            return TF.resize(image, size, self.mode)
        return image


class RandomSquareCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        side_x, side_y = image.size
        max_size = min(side_x, side_y, self.max_size)
        min_size = min(max_size, self.min_size)
        size = torch.randint(min_size, max_size + 1, ()).item()
        offset_x = torch.randint(0, side_x - size + 1, ()).item()
        offset_y = torch.randint(0, side_y - size + 1, ()).item()
        return image.crop((offset_x, offset_y, offset_x + size, offset_y + size))


def get_gaussian_blur_kernels(sigma, radius):
    kernel_size = radius * 2 + 1
    x = torch.linspace(-radius, radius, kernel_size, device=sigma.device) / sigma[:, None]
    kernels_1d = torch.distributions.Normal(0, 1).log_prob(x).exp()
    kernels = (kernels_1d.unsqueeze(2) @ kernels_1d.unsqueeze(1))
    return kernels / kernels.sum(dim=[1, 2], keepdim=True)


def batch_depthwise_conv2d(batch, kernels):
    n, c, h, w = batch.shape
    batch = batch.view([1, n * c, h, w])
    pad_h, pad_w = (kernels.shape[1] - 1) // 2, (kernels.shape[2] - 1) // 2
    batch = F.pad(batch, (pad_w, pad_w, pad_h, pad_h), 'reflect')
    kernels = kernels.to(batch).unsqueeze(1).repeat_interleave(c, 0)
    output = F.conv2d(batch, kernels, groups=n * c)
    return output.view([n, c, h, w])


def random_gaussian_blur(batch):
    sigma = torch.rand(batch.shape[0], device=batch.device) * (0.6 - 0.4) + 0.4
    u = torch.rand(batch.shape[0], device=batch.device)
    kernels = get_gaussian_blur_kernels(sigma, 1)
    blurred = batch_depthwise_conv2d(batch, kernels)
    return torch.where(u[:, None, None, None] < 0.5, blurred, batch)


def prepare_data(batch):
    x = resize_right.resize(batch, 0.25, pad_mode='reflect').clamp(-1, 1).contiguous()
    return random_gaussian_blur(x)


class LightningDiffusion(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DiffusionModel()
        self.model_ema = deepcopy(self.model)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        return self.model_ema(*args, **kwargs)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=1e-5)

    def eval_batch(self, batch):
        reals = batch[0]
        with torch.cuda.amp.autocast(False):
            low_res = prepare_data(reals)

        # Sample timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(reals)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        v = self(noised_reals, t, low_res)
        return F.mse_loss(v, targets)

    def training_step(self, batch, batch_idx):
        loss = self.eval_batch(batch)
        log_dict = {'train/loss': loss.detach()}
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.trainer.global_step < 20000:
            decay = 0.99
        elif self.trainer.global_step < 100000:
            decay = 0.999
        else:
            decay = 0.9999
        ema_update(self.model, self.model_ema, decay)


class DemoCallback(pl.Callback):
    def __init__(self, batch):
        super().__init__()
        self.batch = batch
        self.low_res = prepare_data(batch)

    @rank_zero_only
    @torch.no_grad()
    def on_batch_end(self, trainer, module):
        if trainer.global_step % 2500 != 0:
            return

        batch, low_res = self.batch.to(module.device), self.low_res.to(module.device)
        noise = torch.randn_like(batch)
        with eval_mode(module):
            fakes = sample(module, noise, 1000, 1, {'low_res': low_res})

        low_res_big = resize_right.resize(low_res, out_shape=batch.shape, pad_mode='reflect')
        grid = rearrange([batch, low_res_big, fakes],
                         't (s1 s2) c h w -> c (s2 h) (s1 t w)', s1=3)
        image = TF.to_pil_image(grid.add(1).div(2).clamp(0, 1))
        filename = f'demo_{trainer.global_step:08}.png'
        image.save(filename)
        log_dict = {'demo_grid': wandb.Image(image)}
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-set', type=Path, required=True,
                   help='the training set location')
    args = p.parse_args()

    paths = [path.rstrip() for path in args.train_set.read_text().split('\n')]
    paths = [path for path in paths if path]
    print('Found', len(paths), 'images')

    batch_size = 16

    min_size = 512
    max_size = 512
    size = 192
    # small_size = 192 // 4

    tf = transforms.Compose([
        ToMode('RGB'),
        RandomResizeIfLarger(min_size, max_size, transforms.InterpolationMode.LANCZOS),
        ResizeIfSmaller(size, transforms.InterpolationMode.LANCZOS),
        transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = ImageDataset(paths, transform=tf)
    train_dl = data.DataLoader(train_set, batch_size, shuffle=True,
                               num_workers=12, persistent_workers=True, pin_memory=True)

    demo_dl = data.DataLoader(train_set, 18, shuffle=True)
    demo_batch = next(iter(demo_dl))[0].cuda()

    model = LightningDiffusion()
    wandb_logger = pl.loggers.WandbLogger(project='kat-diffusion')
    wandb_logger.watch(model.model)
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=25000, save_top_k=-1)
    demo_callback = DemoCallback(demo_batch)
    exc_callback = ExceptionCallback()
    trainer = pl.Trainer(
        gpus=8,
        accelerator='ddp',
        precision=16,
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        resume_from_checkpoint='yfcc_upscaler_2_start_1.ckpt',
    )

    trainer.fit(model, train_dl)


if __name__ == '__main__':
    main()
