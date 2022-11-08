#!/usr/bin/env python3

"""Assembles images into a grid."""

import math
import sys

from PIL import Image


def main(image_paths, nrow=None):
    images = [Image.open(image) for image in image_paths]
    mode = images[0].mode
    size = images[0].size
    for image, name in zip(images, image_paths):
        if image.mode != mode:
            print(f'Error: Image {name} had mode {image.mode}, expected {mode}', file=sys.stderr)
            sys.exit(1)
        if image.size != size:
            print(f'Error: Image {name} had size {image.size}, expected {size}', file=sys.stderr)
            sys.exit(1)

    n = len(images)
    x = nrow if nrow else math.ceil(n**0.5)
    y = math.ceil(n / x)

    output = Image.new(mode, (size[0] * x, size[1] * y))
    for i, image in enumerate(images):
        cur_x, cur_y = i % x, i // x
        output.paste(image, (size[0] * cur_x, size[1] * cur_y))

    return(output)


if __name__ == '__main__':
    main()
