import os
import time
import random
import io
import json
import torch
from torch import multiprocessing as mp
import sqlite3
import asyncio
import threading
import queue
import nextcord
from nextcord.ext import commands
from simulacra_glide_sample import main as gen
from yfcc_upscale import main as upscale
from collections import namedtuple

if not os.path.exists("db.sqlite"):
    db = sqlite3.connect('db.sqlite')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE users
                      (id INTEGER PRIMARY KEY, admin INTEGER, banned INTEGER,
                       verified INTEGER, name TEXT)''')
    cursor.execute("INSERT INTO users VALUES (621583764174143488, 1, 0, 1, 'John David Pressman')")
    cursor.execute('''CREATE TABLE generations
                      (id INTEGER PRIMARY KEY, uid INTEGER, mid INTEGER, 
                       method INTEGER, prompt TEXT, 
                       FOREIGN KEY(uid) REFERENCES users(id))''')
    cursor.execute('''CREATE TABLE ratings
                      (uid INTEGER, gid INTEGER, rating INTEGER,
                       FOREIGN KEY(uid) REFERENCES users(id),
                       FOREIGN KEY(gid) REFERENCES generations(id),
                       PRIMARY KEY(uid, gid))''')
    cursor.execute('''CREATE TABLE flags
                      (uid INTEGER, gid INTEGER, 
                       FOREIGN KEY(uid) REFERENCES users(id),
                       FOREIGN KEY(gid) REFERENCES generations(id),
                       PRIMARY KEY(uid, gid))''')
    cursor.execute('''CREATE TABLE upscales
                      (gid INTEGER, choice INTEGER, method INTEGER,
                       FOREIGN KEY(gid) REFERENCES generations(id),
                       PRIMARY KEY(gid, choice))''')
    db.commit()
    cursor.close()

class Users:
    def __init__(self, dbpath):
        self.dbpath = dbpath

    def add_user(self, uid, name, admin=0, verified=1):
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        user = cursor.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                              (uid, admin, 0, verified, name))
        db.commit()
        cursor.close()
        db.close()
        
    def is_user(self, uid):
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        user = cursor.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        if user:
            return user
        else:
            return False

class Generations:
    def __init__(self, dbpath):
        self.dbpath = dbpath
        db = sqlite3.connect(dbpath)
        cursor = db.cursor()
        try:
            self._next_seed = cursor.execute('SELECT id FROM generations ORDER BY id DESC LIMIT 1').fetchone()[0] + 1
        except TypeError:
            self._next_seed = 0

    def get_next_seed(self):
        self._next_seed += 1
        return self._next_seed - 1

    def get_gen_from_mid(self, mid):
        """Get generation data from message ID."""
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        return cursor.execute("SELECT * FROM generations WHERE mid=?", (mid,)).fetchone()

        
class Ratings:
    def __init__(self, dbpath):
        self.dbpath = dbpath
        
    def record_rating(self, uid: int, gid: int, rating: int):
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        cursor.execute("INSERT INTO ratings VALUES (?,?,?)", (uid, gid, rating))
        db.commit()
        db.close()

        
class Flags:
    def __init__(self, dbpath):
        self.dbpath = dbpath

    def record_flag(self, uid: int, gid: int):
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        cursor.execute("INSERT INTO flags VALUES (?,?)", (uid, gid))
        db.commit()
        db.close()


def do_gen_job(messages, interaction, job):
    os.environ['CUDA_VISIBLE_DEVICES'] = job.device
    gen(job)
    messages.put((interaction,job))

def do_upscale_job(messages, interaction, job):
    os.environ['CUDA_VISIBLE_DEVICES'] = job.device
    upscale(job)
    messages.put((interaction,job))
        
        
class Jobs:
    """Job controller for SimulacraBot"""
    def __init__(self, dbpath='db.sqlite'):
        self._gpu_table = [None for i in range(torch.cuda.device_count())]
        self._pending = queue.Queue()
        self._scheduled = {}
        self._pending_messages = mp.Queue()
        self._dbpath = dbpath
        
    async def main(self):
        while 1:
            if not self._pending_messages.empty():
                response = self._pending_messages.get()
                if type(response[1]) == UpscaleJob:
                    query, params = await self.finish_upscale_job(response[0],
                                                                  response[1])
                else:
                    query, params = await self.finish_gen_job(response[0],
                                                              response[1])
                
                db = sqlite3.connect(self._dbpath)
                cursor = db.cursor()
                cursor.execute(query, params)
                db.commit()
                cursor.close()
                db.close()
            if None not in self._gpu_table:
                time.sleep(0.25)
                continue
            # Job dispatch
            try:
                next_job = self._pending.get(block=False)
            except:
                time.sleep(0.25)
                continue
            device_index = self._gpu_table.index(None)
            self._gpu_table[device_index] = time.time()
            interaction, job = self._scheduled[next_job].pop()
            job.device = str(device_index)
            if type(job) == UpscaleJob:
                # Make values picklable
                interaction = {"cid":interaction.channel.id,
                               "mention":interaction.user.mention}
                upscale_job = mp.Process(
                    target=do_upscale_job,
                    args=(self._pending_messages,interaction,job)
                )
                upscale_job.start()
            else:
                # Make values picklable
                interaction = {"cid":interaction.channel.id,
                               "mention":interaction.message.author.mention,
                               "uid":interaction.message.author.id}
                gen_job = mp.Process(
                    target=do_gen_job, 
                    args=(self._pending_messages,interaction,job)
                )
                gen_job.start()

    def submit(self, interaction, job):
        try:
            uid = interaction.user.id
        except AttributeError:
            uid = interaction.message.author
        try:
            self._scheduled[uid]
        except KeyError:
            self._scheduled[uid] = []

        if len(self._scheduled[uid]) < 2:
            self._pending.put(uid)
            self._scheduled[uid].append((interaction, job))
            return True
        else:
            return False
        
    async def finish_gen_job(self, interaction, job):
        view = GenerationButtons()
        embed = nextcord.Embed(title="Feedback", description="")
        embed.add_field(name="Ratings", value=0)
        embed.add_field(name="Flags", value=0)
        upload = nextcord.File(str(job.seed) + "_" + job.prompt.replace(" ", "_").replace("/","_") +
                               "_1" + ".png")
        channel = bot.get_channel(interaction["cid"])
        coroutine = channel.send(
            job.prompt + f" by {interaction['mention']}",
            embed=embed,
            file=upload, view=view)
        response = asyncio.run_coroutine_threadsafe(coroutine, bot.loop).result()
        self._gpu_table[int(job.device)] = None
        return ("INSERT INTO generations VALUES (?, ?, ?, ?, ?)",
                (job.seed, interaction["uid"], response.id, 2, job.prompt))
        
    async def finish_upscale_job(self, interaction, job):
        upload_path = job.input.replace(".png", "_4x_upscale.png")
        upload = nextcord.File(upload_path)
        channel = bot.get_channel(interaction["cid"])
        coroutine = channel.send(
            "'" + job.prompt + f"' upscaled by {interaction['mention']}",
            file=upload
        )
        response = asyncio.run_coroutine_threadsafe(coroutine, bot.loop).result()
        self._gpu_table[int(job.device)] = None
        return ("INSERT INTO upscales VALUES (?,?,?)", (job.gen[0], job.index, 0))
        
TESTING_GUILD_ID = 958788735254822972  # Replace with your guild ID

bot = commands.Bot(command_prefix='.')

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

class UpscaleJob:
    def __init__(self, input, index, checkpoint, eta, output, seed, steps):
        self.input = input
        self.index = index
        self.checkpoint = checkpoint
        self.eta = eta
        self.output = output
        self.seed = seed
        self.steps = steps

class GridButtons(nextcord.ui.View):
    def __init__(self, parent_msg):
        super().__init__(timeout=None)
        self.parent_msg = parent_msg
        self.value = None
    
    async def upscale(self, button, interaction, index):
        gen = generations.get_gen_from_mid(self.parent_msg.id)
        if interaction.user.id != gen[1]:
            await interaction.user.send("Only the user that generated an output can upscale it.")
            return
        prompt = gen[-1]
        img_path = str(gen[0]) + "_" + prompt.replace(" ", "_").replace("/","_") + "_" + str(index) + ".png"
        job = UpscaleJob(input=img_path,
                         index=index,
                         checkpoint="yfcc_upscaler_2.ckpt",
                         eta=1.,
                         output=img_path.replace(".png", "_4x_upscale.png"),
                         seed=gen[0],
                         steps=150)
        job.gen = gen
        job.prompt = prompt
        button.label = "Submitted"
        button.style = nextcord.ButtonStyle.green
        button.disabled = True
        await interaction.message.edit(view=self)
        jobs.submit(interaction, job)

    @nextcord.ui.button(label="U1", custom_id="U1")
    async def U1(self, button, interaction):
        await self.upscale(button, interaction, 1)

    @nextcord.ui.button(label="U2", custom_id="U2")
    async def U2(self, button, interaction):
        await self.upscale(button, interaction, 2)

    @nextcord.ui.button(label="U3", custom_id="U3")
    async def U3(self, button, interaction):
        await self.upscale(button, interaction, 3)

    @nextcord.ui.button(label="U4", custom_id="U4")
    async def U4(self, button, interaction):
        await self.upscale(button, interaction, 4)

    @nextcord.ui.button(label="U5", custom_id="U5")
    async def U5(self, button, interaction):
        await self.upscale(button, interaction, 5)

    @nextcord.ui.button(label="U6", custom_id="U6")
    async def U6(self, button, interaction):
        await self.upscale(button, interaction, 6)

    @nextcord.ui.button(label="U7", custom_id="U7")
    async def U7(self, button, interaction):
        await self.upscale(button, interaction, 7)

    @nextcord.ui.button(label="U8", custom_id="U8")
    async def U8(self, button, interaction):
        await self.upscale(button, interaction, 8)


class GenerationButtons(nextcord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.value = None

    async def rate(self, interaction, rating):
        # TODO: Let users update their rating
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot.")
            return
        message = interaction.message
        generation = generations.get_gen_from_mid(message.id)
        ratings.record_rating(interaction.user.id, generation[0], rating)
        # Update number of ratings
        num_ratings = int(message.embeds[0].fields[0].value)
        num_ratings += 1
        if self.children[15].style != nextcord.ButtonStyle.green:
            self.children[15].disabled = False
        embed = message.embeds[0].set_field_at(0, name="Ratings",
                                               value=num_ratings)
        await message.edit(view=self, embed=embed)
        
    @nextcord.ui.button(label="Worst (1)", custom_id="R1")
    async def R1(self, button, interaction):
        await self.rate(interaction, 1)

    @nextcord.ui.button(label="2", custom_id="R2")
    async def R2(self, button, interaction):
        await self.rate(interaction, 2)

    @nextcord.ui.button(label="3", custom_id="R3")
    async def R3(self, button, interaction):
        await self.rate(interaction, 3)

    @nextcord.ui.button(label="4", custom_id="R4")
    async def R4(self, button, interaction):
        await self.rate(interaction, 4)

    @nextcord.ui.button(label="5", custom_id="R5")
    async def R5(self, button, interaction):
        await self.rate(interaction, 5)

    @nextcord.ui.button(label="6", custom_id="R6")
    async def R6(self, button, interaction):
        await self.rate(interaction, 6)

    @nextcord.ui.button(label="7", custom_id="R7")
    async def R7(self, button, interaction):
        await self.rate(interaction, 7)

    @nextcord.ui.button(label="8", custom_id="R8")
    async def R8(self, button, interaction):
        await self.rate(interaction, 8)

    @nextcord.ui.button(label="9", custom_id="R9")
    async def R9(self, button, interaction):
        await self.rate(interaction, 9)

    @nextcord.ui.button(label="Best (10)", custom_id="R10")
    async def R10(self, button, interaction):
        await self.rate(interaction, 10)

    @nextcord.ui.button(label="---", disabled=True, row=3)
    async def filler1(self, button, interaction):
        pass

    @nextcord.ui.button(label="---", disabled=True, row=3)
    async def filler2(self, button, interaction):
        pass

    @nextcord.ui.button(label="---", disabled=True, row=3)
    async def filler3(self, button, interaction):
        pass

    @nextcord.ui.button(label="---", disabled=True, row=3)
    async def filler4(self, button, interaction):
        pass

    @nextcord.ui.button(label="---", disabled=True, row=3)
    async def filler5(self, button, interaction):
        pass
        
    @nextcord.ui.button(label="Grid", custom_id="grid", disabled=True, row=4)
    async def grid(self, button, interaction):
        gen = generations.get_gen_from_mid(interaction.message.id)
        if interaction.user.id != gen[1]:
            await interaction.user.send("Only the user that generated an output can get its grid.")
            return
        prompt = gen[-1]
        img_path = str(gen[0]) + "_" + prompt.replace(" ", "_").replace("/","_") + "_grid" + ".png"
        upload = nextcord.File(img_path)
        view = GridButtons(interaction.message)
        await interaction.send(
            "'" + prompt + f"' full grid",
            file=upload,
            view=view
        )
        button.style = nextcord.ButtonStyle.green
        button.disabled = True
        await interaction.message.edit(view=self)
        
    @nextcord.ui.button(label="------", custom_id="spacer2", disabled=True,
                        style=nextcord.ButtonStyle.gray, row=4)
    async def spacer2(self, button, interaction):
        pass

    @nextcord.ui.button(label="------", custom_id="spacer3", disabled=True,
                        style=nextcord.ButtonStyle.gray, row=4)
    async def spacer3(self, button, interaction):
        pass
    
    @nextcord.ui.button(label="Flag", custom_id="flag", style=nextcord.ButtonStyle.danger, row=4)
    async def flag(self, button, interaction):
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot.")
            return
        message = interaction.message
        generation = generations.get_gen_from_mid(message.id)
        flags.record_flag(interaction.user.id, generation[0])
        # Update number of flags
        num_flags = int(message.embeds[0].fields[1].value)
        num_flags += 1
        embed = message.embeds[0].set_field_at(1, name="Flags",
                                               value=num_flags)
        await message.edit(embed=embed)


class Job:
    def __init__(self, prompt, cloob_checkpoint, scale, cutn, device, ddim_eta,
                 method, H, W, n_iter, n_samples, seed, ddim_steps, plms):
        self.prompt = prompt
        self.cloob_checkpoint = cloob_checkpoint
        self.scale = scale
        self.cutn = cutn
        self.device = device
        self.ddim_eta = ddim_eta
        self.method = method
        self.H = H
        self.W = W
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.seed = seed
        self.ddim_steps = ddim_steps
        self.plms = plms


#@bot.slash_command(description="My first slash command", guild_ids=[TESTING_GUILD_ID])
@bot.command()
async def add(interaction: nextcord.Interaction):
    if not users.is_user(interaction.message.author.id):
        await interaction.message.author.send("You must agree to the TOS before using SimulacraBot.")
        return
    if interaction.channel.id not in channel_whitelist:
        return
    for word in banned_words:
        if (' ' + word) in interaction.message.content:
            await interaction.send("'{}' is not an allowed word by the NSFW filter".format(word))
            return
    prompt = interaction.message.content.split(".add")[1].strip()
    seed = generations.get_next_seed()
    job = Job(prompt=prompt,
              cloob_checkpoint='cloob_laion_400m_vit_b_16_16_epochs',
              scale=5.,
              cutn=32,
              device='cuda:0',
              ddim_eta=0.,
              method='ddim',
              H=256,
              W=256,
              n_iter=1,
              n_samples=8,
              seed=seed,
              ddim_steps=50,
              plms=True)
    # Unobtrusively let user know we are generating
    success = jobs.submit(interaction, job)
    if success:
        await interaction.message.add_reaction('👍')
        await interaction.message.add_reaction('⏲️')
    else:
        await interaction.message.add_reaction('👎')
        await interaction.message.add_reaction('2️⃣')

@bot.command()
async def export(interaction: nextcord.Interaction):
    if type(interaction.channel) != nextcord.channel.DMChannel:
        return
    else:
        export = {}
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        export['user'] = cursor.execute('SELECT * FROM users where id=?',
                                        (interaction.message.author.id,)).fetchone()
        export['generations'] = cursor.execute('SELECT * FROM generations where uid=?',
                                               (interaction.message.author.id,)).fetchall()
        export['ratings'] = cursor.execute('SELECT * FROM ratings where uid=?',
                                           (interaction.message.author.id,)).fetchall()
        export['flags'] = cursor.execute('SELECT * FROM flags where uid=?',
                                         (interaction.message.author.id,)).fetchall()
        export['upscales'] = cursor.execute('SELECT * FROM upscales where uid=?',
                                            (interaction.message.author.id,)).fetchall()
        db.close()
        upload = nextcord.File(io.StringIO(json.dumps(export)),
                               filename="{}_export_{}.json".format(
                                   interaction.message.author.id,
                                   int(time.time())))
        await interaction.send("Your SimulacraBot text data:\n\n(Due to their size images must be manually requested by sending mail to gdpr@stability.ai or pinging bot admins such as Drexler#4006)" , file=upload)

@bot.command()
async def adduser(interaction: nextcord.Interaction):
    user = users.is_user(interaction.message.author.id)
    if (not user) or (not user[1]):
        interaction.message.author.send("This command is restricted to admins only.")
    else:
        uid, name, verified = interaction.message.content.split(".adduser")[1].strip().split(",")
        users.add_user(int(uid), name, verified=int(verified))
        if int(verified):
            await interaction.message.author.send(f"Added '{name}' as verified user.")
        else:
            await interaction.message.author.send(f"Added '{name}' as normal user.")

if __name__ == '__main__' :
            
    with open('channel_whitelist.txt') as infile:
        channel_whitelist = [int(channel.strip()) for channel in infile.readlines()]
        
    with open('banned_words.txt') as infile:
        banned_words = [word.strip() for word in infile.readlines()]
    
    with open('token.txt') as infile:
        token = infile.read().strip()

    mp.set_start_method('spawn')
    
    users = Users('db.sqlite')
    generations = Generations('db.sqlite')
    ratings = Ratings('db.sqlite')
    flags = Flags('db.sqlite')
    jobs = Jobs('db.sqlite')

    jobs_thread = threading.Thread(
        target=asyncio.run,
        args=(jobs.main(),)
    )
    jobs_thread.start()

    bot.run(token)

