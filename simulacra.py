import os
import time
import secrets
import io
import json
import torch
from torch import multiprocessing as mp
import logging
import sqlite3
import asyncio
import threading
import queue
import nextcord
from nextcord.ext import commands
from simulacra_glide_sample import main as gen
from yfcc_upscale import main as upscale
from collections import namedtuple

intents = nextcord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='.', intents=intents)

logging.basicConfig(filename='simulacra.log', encoding='utf-8', level=logging.WARNING)

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
    cursor.execute('''CREATE TABLE images
                           (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           gid INTEGER,
                           idx INTEGER,
                           FOREIGN KEY(gid) REFERENCES generations(id),
                           UNIQUE(gid,idx))''')
    cursor.execute('''CREATE TABLE ratings
                           (uid INTEGER, iid INTEGER, rating INTEGER,
                           FOREIGN KEY(uid) REFERENCES users(id),
                           FOREIGN KEY(iid) REFERENCES images(id),
                           PRIMARY KEY(uid, iid))''')
    cursor.execute('''CREATE TABLE flags
                           (uid INTEGER, iid INTEGER, 
                           FOREIGN KEY(uid) REFERENCES users(id),
                           FOREIGN KEY(iid) REFERENCES images(id),
                           PRIMARY KEY(uid, iid))''')
    cursor.execute('''CREATE TABLE upscales
                           (iid INTEGER, method INTEGER,
                           FOREIGN KEY(iid) REFERENCES images(id),
                           PRIMARY KEY(iid))''')
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
        
    def record_rating(self, uid: int, gid: int, rating: int, index: int = 1):
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        cursor.execute("SELECT id FROM images WHERE gid=? AND idx=?",
                       (gid, index))
        image_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO ratings VALUES (?,?,?)", (uid, image_id, rating))
        db.commit()
        db.close()

        
class Flags:
    def __init__(self, dbpath):
        self.dbpath = dbpath

    def get_flags(self):
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        cursor.execute("SELECT * FROM flags;")
        return cursor.fetchall()
        
    def record_flag(self, uid: int, gid: int, index: int = 1):
        db = sqlite3.connect(self.dbpath)
        cursor = db.cursor()
        cursor.execute("SELECT id FROM images WHERE gid=? AND idx=?",
                       (gid, index))
        image_id = cursor.fetchone()[0]
        cursor.execute("INSERT INTO flags VALUES (?,?)", (uid, image_id))
        db.commit()
        db.close()


def do_gen_job(messages, interaction, job):
    gen(job)
    messages.put((interaction,job))

def do_upscale_job(messages, interaction, job):
    upscale(job)
    messages.put((interaction,job))
        
        
class Jobs:
    """Job controller for SimulacraBot"""
    def __init__(self, channels, dbpath='db.sqlite'):
        self._gpu_table = [None for i in range(torch.cuda.device_count())]
        self._pending = queue.Queue()
        self._scheduled = {}
        self._pending_messages = mp.Queue()
        self._dbpath = dbpath
        self._channels = channels
        
    async def main(self):
        last_rate_prompt = time.time()
        while 1:
            if not self._pending_messages.empty():
                db = sqlite3.connect(self._dbpath)
                cursor = db.cursor()
                response = self._pending_messages.get()
                if type(response[1]) == UpscaleJob:
                    try:
                        query, params = await self.finish_upscale_job(response[0],
                                                                      response[1])
                        cursor.execute(query, params)
                    except sqlite3.IntegrityError as e:
                        logging.warning("Duplicate upscale occurred - {}".format(str(e)))
                        cursor.close()
                        db.close()
                        continue
                # TODO: Change this class to 'GenJob' or some such
                elif type(response[1]) == Job:
                    query, params = await self.finish_gen_job(response[0],
                                                              response[1])
                    cursor.execute(query, params)
                    for i in range(1,9):
                        cursor.execute("INSERT INTO images(gid, idx) VALUES (?,?)",
                                       (params[0], i))
                db.commit()
                cursor.close()
                db.close()
            if (time.time() - last_rate_prompt) >= 60:
                await self.rate_prompt()
                last_rate_prompt = time.time()
            if None not in self._gpu_table:
                for job_time in enumerate(self._gpu_table):
                    # Clear out stuck job registers/timeout                   
                    i,t = job_time
                    if (time.time() - t) >= 120:
                        self._gpu_table[i] = None
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
            os.environ['CUDA_VISIBLE_DEVICES'] = job.device
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

    async def rate_prompt(self):
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        # Try finding images with no rating
        cursor.execute('''SELECT generations.id, images.id, prompt, images.idx, COUNT(rating) as num_ratings FROM 
                       generations LEFT OUTER JOIN images ON images.gid=generations.id
                       LEFT OUTER JOIN ratings ON images.id=ratings.iid 
                       GROUP BY images.id 
                       HAVING num_ratings == 0;''')
        to_rate = cursor.fetchall()
        for cid in self._channels:
            try:
                gen = secrets.choice(to_rate)
            except IndexError:
                return
            view = RatingButtons(gen[0], index=gen[3])
            embed = nextcord.Embed(title="Feedback", description="")
            cursor.execute("SELECT COUNT(*) from ratings where iid=?", (gen[1],))
            ratings = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) from flags where iid=?", (gen[1],))
            flags = cursor.fetchone()[0]
            embed.add_field(name="Ratings", value=ratings)
            embed.add_field(name="Flags", value=flags)
            upload = nextcord.File(str(gen[0]) + "_" + gen[2].replace(" ", "_").replace("/","_") +
                               "_" + str(gen[3]) + ".png")
            channel = bot.get_channel(cid)
            coroutine = channel.send(
                gen[2],
                file=upload,
                embed=embed,
                view=view)
            asyncio.run_coroutine_threadsafe(coroutine, bot.loop).result()
                
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
        return ("INSERT INTO upscales VALUES (?,?)", (job.iid, 0))


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

class UpscaleJob:
    def __init__(self, input, iid, checkpoint, eta, output, seed, steps):
        self.input = input
        self.iid = iid
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
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT id FROM images WHERE gid=? AND idx=?",
                       (gen[0], index))
        image_id = cursor.fetchone()[0]
        assert image_id
        cursor.close()
        db.close()
        job = UpscaleJob(input=img_path,
                         iid=image_id,
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


class AbstractButtons(nextcord.ui.View):
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


class GenerationButtons(AbstractButtons):
    def __init__(self):
        super().__init__()
        
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

class RatingButtons(AbstractButtons):
    def __init__(self, gid, index: int = 1):
        super().__init__()
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("select * from generations where id=?", (gid,))
        self.generation = cursor.fetchone()
        self.index = index
        cursor.close()
        db.close()

    async def rate(self, interaction, rating):
        # TODO: Let users update their rating
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot.")
            return
        message = interaction.message
        generation = self.generation
        ratings.record_rating(interaction.user.id, generation[0], rating, index = self.index)
        # Update number of ratings
        num_ratings = int(message.embeds[0].fields[0].value)
        num_ratings += 1
        embed = message.embeds[0].set_field_at(0, name="Ratings",
                                               value=num_ratings)
        await message.edit(view=self, embed=embed)

    @nextcord.ui.button(label="Flag", custom_id="flag", style=nextcord.ButtonStyle.danger, row=4)
    async def flag(self, button, interaction):
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot.")
            return
        message = interaction.message
        generation = self.generation
        flags.record_flag(interaction.user.id, generation[0], index = self.index)
        # Update number of flags
        num_flags = int(message.embeds[0].fields[1].value)
        num_flags += 1
        embed = message.embeds[0].set_field_at(1, name="Flags",
                                               value=num_flags)
        await message.edit(embed=embed)

class WarningSelect(nextcord.ui.View):
    def __init__(self, generation, recipient):
        super().__init__(timeout=None)
        self.value = None
        self.generation = generation
        self.recipient = recipient

    @nextcord.ui.select(placeholder="Why is this gen being removed?",
                        custom_id="warnselect",
                        options=[
                            nextcord.SelectOption(label="NSFW",
                                                  value="was NSFW"),
                            nextcord.SelectOption(label="Hateful",
                                                  value="was hateful"),
                            nextcord.SelectOption(label="Personal Information",
                                                  value="included personal information"),
                            nextcord.SelectOption(label="Copyright/Trademarks",
                                                  value="violated copyright"),
                        ])
    async def warning(self, select, interaction):
        await self.recipient.send(f"Your generation '{self.generation[-1]}' was "
                                  f"removed from Simulacra Aesthetic captions because it {select.values[0]}.")
        
class ModerationButtons(nextcord.ui.View):
    def __init__(self, gid):
        super().__init__(timeout=None)
        self.value = None
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT * FROM generations WHERE id=?", (gid,))
        self.generation = cursor.fetchone()
        cursor.close()
        db.close()

    @nextcord.ui.button(label="OK", custom_id="ok")
    async def ok(self, button, interaction):
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("""DELETE FROM flags WHERE flags.iid IN
                          (SELECT id FROM images WHERE gid=?)""",
                       (self.generation[0],))
        db.commit()
        cursor.close()
        db.close()
        
    @nextcord.ui.button(label="Warn", custom_id="warn")
    async def warn(self, button, interaction):
        """Send a user a notification that their submission was unacceptable."""
        user = await bot.fetch_user(self.generation[1])
        view = WarningSelect(self.generation, user)
        await interaction.user.send("",
                              view=view)

                
    @nextcord.ui.button(label="Purge", custom_id="purge")
    async def purge(self, button, interaction):
        """Purge a submission from the database."""
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT idx FROM images WHERE gid=?", (self.generation[0],))
        batch_indexes = [x[0] for x in cursor.fetchall()]
        for index in batch_indexes:
            prompt = self.generation[-1]
            img_path = str(self.generation[0]) + "_" + prompt.replace(" ", "_").replace("/","_") + "_" + str(index) + ".png"
            os.remove(img_path)
            try:
                os.remove(img_path.replace(".png", "_4x_upscale.png"))
            except FileNotFoundError:
                continue
        prompt = self.generation[-1]
        img_path = str(self.generation[0]) + "_" + prompt.replace(" ", "_").replace("/","_") + "_grid" + ".png"
        os.remove(img_path)
        cursor.execute("""DELETE FROM upscales WHERE upscales.iid IN
                          (SELECT id FROM images WHERE gid=?)""",
                       (self.generation[0],))
        cursor.execute("""DELETE FROM ratings WHERE ratings.iid IN
                          (SELECT id FROM images WHERE gid=?)""",
                       (self.generation[0],))
        cursor.execute("""DELETE FROM flags WHERE flags.iid IN
                          (SELECT id FROM images WHERE gid=?)""",
                       (self.generation[0],))
        cursor.execute("DELETE FROM images WHERE gid=?", (self.generation[0],))
        cursor.execute("DELETE FROM generations WHERE id=?",
                       (self.generation[0],))
        db.commit()
        cursor.close()
        db.close()
        await interaction.user.send("Purged!")

    @nextcord.ui.button(label="Ban", custom_id="ban")
    async def ban(self, button, interaction):
        """Ban a user for submitting a prompt."""
        pass
        
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
async def rate(interaction: nextcord.Interaction):
    if type(interaction.channel) != nextcord.channel.DMChannel:
        return

    db = sqlite3.connect('db.sqlite')
    cursor = db.cursor()
    # Try finding images with no rating
    cursor.execute('''SELECT generations.id, images.id, prompt, images.idx, COUNT(rating) as num_ratings FROM 
                      generations LEFT OUTER JOIN images ON images.gid=generations.id
                      LEFT OUTER JOIN ratings ON images.id=ratings.iid 
                      GROUP BY images.id 
                      HAVING num_ratings == 0;''')
    to_rate = cursor.fetchall()[:5]
    # If all images rated find images with too few ratings
    #if not to_rate:
    #    cursor.execute('''SELECT id, prompt, COUNT(rating) as num_ratings FROM 
    #                      images JOIN ratings ON images.id=ratings.iid 
    #                      WHERE images.id NOT IN 
    #                      (SELECT iid FROM ratings WHERE uid=?) 
    #                      GROUP BY images.id HAVING num_ratings < 3;''',
    #                   (interaction.message.author.id,))
    #    to_rate = cursor.fetchall()[:5]
    for gen in to_rate:
        view = RatingButtons(gen[0], index=gen[3])
        embed = nextcord.Embed(title="Feedback", description="")
        cursor.execute("SELECT COUNT(*) from ratings where iid=?", (gen[1],))
        ratings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) from flags where iid=?", (gen[1],))
        flags = cursor.fetchone()[0]
        embed.add_field(name="Ratings", value=ratings)
        embed.add_field(name="Flags", value=flags)
        upload = nextcord.File(str(gen[0]) + "_" + gen[2].replace(" ", "_").replace("/","_") +
                               "_" + str(gen[3]) + ".png")
        
        await interaction.channel.send(
            gen[2],
            file=upload,
            embed=embed,
            view=view)
    cursor.close()
    db.close()
            
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
        export['images'] = cursor.execute('''SELECT images.* FROM images INNER JOIN generations ON
                                          images.gid=generations.id where generations.uid=?''',
                                          (interaction.message.author.id,)).fetchall()
        export['ratings'] = cursor.execute('SELECT * FROM ratings where uid=?',
                                           (interaction.message.author.id,)).fetchall()
        export['flags'] = cursor.execute('SELECT * FROM flags where uid=?',
                                         (interaction.message.author.id,)).fetchall()
        export['upscales'] = cursor.execute('''SELECT upscales.* FROM upscales 
                                            INNER JOIN images ON 
                                            images.id=upscales.iid INNER JOIN
                                            generations ON generations.id=images.gid 
                                            where generations.uid=?''',
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

@bot.command()
async def mod(interaction: nextcord.Interaction):
    user = users.is_user(interaction.message.author.id)
    if type(interaction.channel) != nextcord.channel.DMChannel:
        return
    if (not user) or (not user[1]):
        interaction.message.author.send("This command is restricted to admins only.")
    else:
        to_review = flags.get_flags()[0]
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT * FROM images WHERE id=?", (to_review[1],))
        image = cursor.fetchone()
        cursor.execute("SELECT * FROM generations WHERE id=?", (image[1],))
        gen = cursor.fetchone()
        prompt = gen[-1]
        img_path = str(gen[0]) + "_" + prompt.replace(" ", "_").replace("/","_") + "_grid" + ".png"
        upload = nextcord.File(img_path)
        view = ModerationButtons(image[1])
        await interaction.message.author.send(
            prompt,
            view=view,
            file=upload)
            
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
    jobs = Jobs(channel_whitelist, 'db.sqlite')

    jobs_thread = threading.Thread(
        target=asyncio.run,
        args=(jobs.main(),)
    )
    jobs_thread.start()
    
    bot.run(token)

