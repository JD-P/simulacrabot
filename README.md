# Simulacra Aesthetic Captions Bot

## Introduction

This discord bot is used for collecting the Simulacra Aesthetic Captions dataset, a
public domain collection of prompts, AI generated imagery, and human aesthetic feedback
for use in training [Instruct-like](https://arxiv.org/abs/2203.02155) aesthetic models
to guide AI generated imagery into better satisfying human preferences. The dataset
could also be used to train a prompt generator or recommendation system.

## Installation

The setup for SimulacraBot is unapologetically janky and I have no plans to make it cleaner.

First we git clone this repository and set up our virtual environment.

```
git clone --recursive https://github.com/JD-P/simulacrabot
cd simulacrabot/
python3 -m venv env_simulacra
```

Then we pip install our dependencies.

```
pip3 install nextcord omegaconf resize-right transformers
```

Next we pip install OpenAI's CLIP reposiitory (already cloned during the
recursive clone of this repo).

```
cd CLIP
pip3 install .
cd ..
```

We download the YFCC upscaler SimulacraBot uses to upscale its outputs. This
upscaler is trained by [RiversHaveWings](https://github.com/crowsonkb/) and hasn't
been previously released because it typically produces horribly artifacted upscales,
often ruining the image it's applied to. It is also somehow better than every other
public upscaler I've tried so enjoy the upgrade I guess.

```
wget https://the-eye.eu/public/AI/models/yfcc_upscaler_2.ckpt
sha256sum yfcc_upscaler_2.ckpt 
52a702409d76cd648be55a33de42aab188d676d9bfd23b5ef2ab5a8588c673c0  yfcc_upscaler_2.ckpt
```

We also download the [CompVis latent GLIDE model](https://github.com/crowsonkb/latent-diffusion)
so we can generate outputs to upscale in the first place.

```
mkdir -p latent-diffusion/models/ldm/text2img-large/
wget -O latent-diffusion/models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

Now you need to put your discord bot token, list of banned words, and channel
whitelist into the filepaths below. [You can set up your discord bot
here](https://discord.com/developers/applications) if you're logged in.

```
nano token.txt
nano banned_words.txt
nano channel_whitelist.txt
```

Each file is a single entry on each line like so:

```
my
banned
words
```

The token file is a single line, while `banned_words.txt` and `channel_whitelist.txt`
are multiple lines. In the future this will probably all be replaced with a YAML or JSON
config file.

Then one last thing. We symbolic link some subdirectories rather than try to fight
the python import system.

```
ln -s latent-diffusion/models/ ./models
ln -s latent-diffusion/configs/ ./configs
ln -s taming-transformers/taming ./taming
```

You should now be able to boot the bot.

```
python3 simulacra.py
```

If all goes as planned it should come online in your server after you invite it
and download a bunch of stuff the first time it tries to generate an output. After
that it should work normally.