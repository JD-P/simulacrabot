#!/usr/bin/env python3

import sys
import argparse
from PIL import Image
import torch
from torchvision.transforms import functional as TF

import yfcc_upscaler_2 as training

sys.path.append("./v-diffusion-pytorch")
from diffusion import sampling, utils

def main(args):
    p = argparse.ArgumentParser()
    p.add_argument('input', type=str,
                   help='the input image')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='the checkpoint to use')
    p.add_argument('--eta', type=float, default=1.,
                   help='the amount of noise to add during sampling (0-1)')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output image')
    p.add_argument('--seed', type=int, default=0,
                   help='the random seed')
    # p.add_argument('--size', '-s', type=int, nargs=2, default=(512, 512),
    #                help='the output size')
    p.add_argument('--steps', type=int, default=100,
                   help='the number of timesteps')
    p.add_argument('--device', type=str, default='cuda:0',
                   help='which CUDA device to use for the upscale')
    #args = p.parse_args()

    model_parent = training.LightningDiffusion.load_from_checkpoint(args.checkpoint)
    model = model_parent.model_ema
    del model_parent
    model = model.half().cuda().eval().requires_grad_(False)

    low_res_pil = Image.open(args.input).convert('RGB')
    low_res = TF.to_tensor(low_res_pil).cuda()[None] * 2 - 1

    side_x, side_y = low_res_pil.size[0] * 4, low_res_pil.size[1] * 4

    torch.manual_seed(args.seed)
    noise = torch.randn([1, 3, side_y, side_x], device=args.device)
    t = torch.linspace(1, 0, args.steps + 1, device=args.device)[:-1]
    steps = utils.get_spliced_ddpm_cosine_schedule(t)
    outs = sampling.plms2_sample(model, noise, steps, {'low_res': low_res})
    outs = outs.add(1).div(2).clamp(0, 1)
    TF.to_pil_image(outs[0]).save(args.output)


if __name__ == '__main__':
    main()
