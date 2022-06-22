import os
import time
import secrets
import io
import json
import torch
from torch import multiprocessing as mp
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
import logging
import sqlite3
import asyncio
import threading
import queue
import nextcord
from nextcord.ext import commands
from simulacra_imagen_sample import main as gen # TODO: Change this to not-gen
# Global conflicts with other variables
from yfcc_upscale import main as upscale
from collections import namedtuple

intents = nextcord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix='.', intents=intents)

logging.basicConfig(filename='simulacra.log', encoding='utf-8', level=logging.WARNING)

class ToMode:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

if not os.path.exists("db.sqlite"):
    db = sqlite3.connect('db.sqlite')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE users
                      (id INTEGER PRIMARY KEY, admin INTEGER, banned INTEGER,
                       verified INTEGER, name TEXT)''')
    cursor.execute('''CREATE TABLE survey
                      (uid INTEGER, qid INTEGER, rating INTEGER,
                      FOREIGN KEY(uid) REFERENCES users(id)
                      PRIMARY KEY(uid, qid))''')
    # cursor.execute("INSERT INTO users VALUES (621583764174143488, 1, 0, 1)")
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
        survey = cursor.execute("SELECT COUNT(*) FROM survey WHERE uid=?", (uid,)).fetchone()[0]
        if user and (survey >= 20):
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
                    for i in range(1,5):
                        cursor.execute("INSERT INTO images(gid, idx) VALUES (?,?)",
                                       (params[0], i))
                db.commit()
                cursor.close()
                db.close()
            if (time.time() - last_rate_prompt) >= 120:
                await self.rate_prompt()
                last_rate_prompt = time.time()
            if None not in self._gpu_table:
                for job_time in enumerate(self._gpu_table):
                    # Clear out stuck job registers/timeout                   
                    i,t = job_time
                    if (time.time() - t) >= 300:
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
                (job.seed, interaction["uid"], response.id, 3, job.prompt))
        
    async def finish_upscale_job(self, interaction, job):
        upload_path = job.input.replace(".png", "_4x_upscale.png")
        # Crunch down upscale so Discord won't choke
        img = Image.open(upload_path)
        to_rgb = ToMode('RGB')
        resize = transforms.Resize(1600, interpolation=transforms.InterpolationMode.LANCZOS)
        img = resize(to_rgb(img))
        img.save(upload_path)
        upload = nextcord.File(upload_path)
        channel = bot.get_channel(interaction["cid"])
        coroutine = channel.send(
            "'" + job.prompt + f"' upscaled by {interaction['mention']}",
            file=upload
        )
        try:
            response = asyncio.run_coroutine_threadsafe(coroutine, bot.loop).result()
        except nextcord.errors.HTTPException:
            upload = nextcord.File(job.input)
            coroutine = channel.send(f"{interaction['mention']}, "
                                     "Your upscale was too large to attach "
                                     "even after downscale to 1600x1600."
                                     " Here's the original.",
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
    def __init__(self, gen):
        super().__init__(timeout=None)
        self.gen = gen
        self.value = None
    
    async def upscale(self, button, interaction, index):
        gen = self.gen
        if interaction.user.id != gen[1]:
            await interaction.user.send("Only the user that generated an output "
                                        "can upscale it. This output was generated"
                                        f" by UID {gen[1]} and you are "
                                        f"{interaction.user.id}.")
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


class AbstractButtons(nextcord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.value = None

    async def rate(self, interaction, rating):
        # TODO: Let users update their rating
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
            return
        message = interaction.message
        generation = generations.get_gen_from_mid(message.id)
        assert generation[-1] == message.content[:len(generation[-1])]
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
        
    @nextcord.ui.button(label="Batch", custom_id="batch", disabled=True, row=4)
    async def batch(self, button, interaction):
        gid = int(interaction.message.attachments[0].filename.split("_")[0])
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT * FROM generations WHERE id=?",
                       (gid,))
        gen = cursor.fetchone()
        cursor.close()
        db.close()
        prompt = gen[-1]
        upload = nextcord.File(str(gen[0]) + "_" + gen[-1].replace(" ", "_").replace("/","_") +
                           "_2" + ".png")
        view = BatchRateStream(gen[0], [2,3,4])
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT * FROM images WHERE gid=?", (gen[0],))
        image_id = cursor.fetchall()[1][0]
        cursor.execute("SELECT COUNT(*) from ratings where iid=?", (image_id,))
        ratings_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) from flags where iid=?", (image_id,))
        flags = cursor.fetchone()[0]
        cursor.close()
        db.close()
        embed = nextcord.Embed(title="Feedback", description="")
        embed.add_field(name="Ratings", value=ratings_count)
        embed.add_field(name="Flags", value=flags)
        await interaction.response.send_message(
            "2. " + prompt,
            file=upload,
            view=view,
            embed=embed,
            ephemeral=True
        )
        button.style = nextcord.ButtonStyle.green
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
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
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
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
            return
        message = interaction.message
        generation = self.generation
        assert generation[-1] == message.content
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
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
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
        
        
class StreamRatingButtons(AbstractButtons):
    def __init__(self):
        super().__init__()
        gen, ratings, flags = self.get_next_image_to_rate()
        gid = gen[0]
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("select * from generations where id=?", (gid,))
        self.generation = cursor.fetchone()
        self.index = gen[3]
        cursor.close()
        db.close()
        

    def get_next_image_to_rate(self):
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        # Try finding images with no rating
        cursor.execute('''SELECT generations.id, images.id, prompt, images.idx, COUNT(rating) as num_ratings FROM 
                          generations LEFT OUTER JOIN images ON images.gid=generations.id
                          LEFT OUTER JOIN ratings ON images.id=ratings.iid 
                          GROUP BY images.id 
                          HAVING num_ratings == 0;''')
        gen = secrets.choice(cursor.fetchall())
        cursor.execute("SELECT COUNT(*) from ratings where iid=?", (gen[1],))
        ratings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) from flags where iid=?", (gen[1],))
        flags = cursor.fetchone()[0]
        cursor.close()
        db.close()
        return gen, ratings, flags
        
    async def rate(self, interaction, rating):
        # TODO: Let users update their rating
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
            return
        message = interaction.message
        generation = self.generation
        try:
            assert generation[-1] == message.content
        except:
            await interaction.user.send("Rating session desynced. Current prompt is "
                                        f"'{message.content}' but rating "
                                        f"is for '{generation[-1]}'.")
            return
        ratings.record_rating(interaction.user.id, generation[0], rating, index = self.index)
        # Update number of ratings
        num_ratings = int(message.embeds[0].fields[0].value)
        num_ratings += 1
        embed = message.embeds[0].set_field_at(0, name="Ratings",
                                               value=num_ratings)
        await message.edit(view=self, embed=embed)
        # Send next message in the chain
        gen, ratings_count, flags = self.get_next_image_to_rate()
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("select * from generations where id=?", (gen[0],))
        self.generation = cursor.fetchone()
        self.index = gen[3]
        cursor.close()
        db.close()
        embed = nextcord.Embed(title="Feedback", description="")
        embed.add_field(name="Ratings", value=ratings_count)
        embed.add_field(name="Flags", value=flags)
        upload = nextcord.File(str(gen[0]) + "_" + gen[2].replace(" ", "_").replace("/","_") +
                           "_" + str(gen[3]) + ".png")
        
        await interaction.user.send(
            gen[2],
            file=upload,
            embed=embed,
            view=self)

    @nextcord.ui.button(label="Flag", custom_id="flag", style=nextcord.ButtonStyle.danger, row=4)
    async def flag(self, button, interaction):
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
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


class BatchRateStream(AbstractButtons):
    def __init__(self, gid, indices):
        super().__init__()
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT * FROM images WHERE gid=?", (gid,))
        image_ids_ = cursor.fetchall()
        cursor.execute("SELECT * FROM generations WHERE id=?", (gid,))
        gen = cursor.fetchone()
        self.gen = gen
        self.image_ids = []
        # Let us restrict the batch to things that need ratings
        for index in indices:
            self.image_ids.append(image_ids_[index-1])
        self.image_ids = [i for i in reversed(self.image_ids)]
        
    async def rate(self, interaction, rating):
        # TODO: Let users update their rating
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
            return
        message = interaction.message
        generation = self.gen
        index = self.image_ids.pop()
        # Make sure our ratings are in sync
        assert (f"{index[2]}. " + generation[-1]) == message.content
        ratings.record_rating(interaction.user.id, generation[0], rating, index = index[2])
        if not self.image_ids: # End condition
            return
        
        # Send next message in the chain
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) from ratings where iid=?", (self.image_ids[-1][0],))
        ratings_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) from flags where iid=?", (self.image_ids[-1][0],))
        flags = cursor.fetchone()[0]
        cursor.close()
        db.close()
        embed = nextcord.Embed(title="Feedback", description="")
        embed.add_field(name="Ratings", value=ratings_count)
        embed.add_field(name="Flags", value=flags)
        upload = nextcord.File(str(generation[0]) + "_" + generation[-1].replace(" ", "_").replace("/","_") +
                           "_" + str(self.image_ids[-1][2]) + ".png")
        
        await interaction.response.send_message(
            f"{self.image_ids[-1][2]}. " + generation[-1],
            file=upload,
            embed=embed,
            view=self,
            ephemeral=True)

    @nextcord.ui.button(label="Flag", custom_id="flag", style=nextcord.ButtonStyle.danger, row=4)
    async def flag(self, button, interaction):
        if not users.is_user(interaction.user.id):
            await interaction.user.send("You must agree to the TOS before using SimulacraBot. Type .signup")
            return
        message = interaction.message
        flags.record_flag(interaction.user.id, self.gen[0], index = self.image_ids[-1][2])
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

class RatingSurvey(AbstractButtons):
    def __init__(self):
        super().__init__()
        self.survey_images = [
            ("The West Coastline. #west #ocean", "survey/6042_The_West_Coastline._west_ocean_6.png"),
            ("hyperrealistic bokeh portrait cinematic matte painting concept art "
             "african american New Orleans",
             "survey/7413_hyperrealistic_bokeh_portrait_cinematic_matte_painting"
             "_concept_art_african_american_New_Orleans_1.png"),
            ("A painting of a swamp at low tide. There is a small portion of a "
             "castle and a patch of blue sky in the far off distance.",
             "survey/4904_A_painting_of_a_swamp_at_low_tide._There_is_a_small_portion_of_"
             "a_castle_and_a_patch_of_blue_sky_in_the_far_off_distance._3.png"),
            ("fantasy illustration of a river of blood, flowing through a field of bones.",
             "survey/549_fantasy_illustration_of_a_river_of_blood_flowing_through_a_field_of_bones._6.png"),
            ("full color inkscape digital art of a hackers lair bedroom computer setup battlestation",
             "survey/1279_full_color_inkscape_digital_art_of_a_hackers_lair_bedroom_computer_setup_battlestation_5.png"),
            ("traditional dark Hijab Muslim Woman Islam Girl Portrait Fashion "
             "People Model Eyes oil on canvas",
             "survey/7435_traditional_dark_Hijab_Muslim_Woman_Islam_Girl_Portrait"
             "_Fashion_People_Model_Eyes_oil_on_canvas_3.png"),
            ("accurate beautiful cute anime symmetric danbooru chibi depiction of a magical girl",
             "survey/6641_accurate_beautiful_cute_anime_symmetric_danbooru_chibi_depiction_of_a_magical_girl_2.png"),
            ("A portrait painting of Adolf Hitler",
             "survey/Der_Führer_Gemälde_Portrait_painting_of_Adolf_Hitler_by_Heinrich_Knirr"
             "_1937_No_known_copyright_restrictions_(artist_died_in_1944)_Imperial_War_Museum_London.png"),
            ("detailed accurate krita digital masterpiece commissioned street "
             "scene of a cyberpunk Tokyo market stall, artstation matte painting",
             "survey/6625_detailed_accurate_krita_digital_masterpiece_commissioned_street_"
             "scene_of_a_cyberpunk_Tokyo_market_stall_artstation_matte_painting_1.png"),
            ("professional HDR flambient golden hour real estate photography of"
             "futuristic hyperrealism scifi fantasy industrial corporate agents",
             "survey/3690_professional_HDR_flambient_golden_hour_real_estate_photography_of"
             "_futuristic_hyperrealism_scifi_fantasy_industrial_corporate_agents_5.png"),
            ("The colors in this illustration are simply gorgeous! They perfectly "
             "compliment the scene and add an extra layer of beauty to the already stunning image!",
             "survey/6424_The_colors_in_this_illustration_are_simply_gorgeous_They_perfectly"
             "_compliment_the_scene_and_add_an_extra_layer_of_beauty_to_the_already_stunning_image_6.png"),
            ("hyperrealism chiaroscuro cinematic oil on canvas matte painting of "
             "professional golden hour flambient real estate photo scifi traditional "
             "japanese onsen shinto temple zen garden",
             "survey/3647_hyperrealism_chiaroscuro_cinematic_oil_on_canvas_matte_painting"
             "_of_professional_golden_hour_flambient_real_estate_photo_scifi_"
             "traditional_japanese_onsen_shinto_temple_zen_garden_4.png"),
            ("I found this abstract art and don't know what it means",
             "survey/3226_I_found_this_abstract_art_and_dont_know_what_it_means_4.png"),
            ("A painting of a mystical creature in a magical forest",
             "survey/585_A_painting_of_a_mystical_creature_in_a_magical_forest_2.png"),
            ("digital painting of a mansion in the middle of a forest, 4k, Ross "
             "Tran, Gurney, Frank Frazetta, Skeeva, Gal Barkan, matayosi, high "
             "fantasy concept key art",
             "survey/4220_digital_painting_of_a_mansion_in_the_middle_of_a_forest_4k_Ross"
             "_Tran_Gurney_Frank_Frazetta_Skeeva_Gal_Barkan_matayosi_high_fantasy_"
             "concept_key_art_6.png"),
            ("professional HDR flambient wide angle bokeh photography of a cute "
             "pink glittery magical fairy enchanted forest fantasy butterflies and poodles",
             "survey/6609_professional_HDR_flambient_wide_angle_bokeh_photography_of_a_"
             "cute_pink_glittery_magical_fairy_enchanted_forest_fantasy_butterflies_and_poodles_1.png"),
            ("chiaroscuro professional oil on canvas matte painting concept art "
             "of a masculine warrior from myths and legends",
             "survey/6548_chiaroscuro_professional_oil_on_canvas_matte_painting_"
             "concept_art_of_a_masculine_warrior_from_myths_and_legends_2.png"),
            ("A nightmarish creature made of shadows and teeth.",
             "survey/1039_A_nightmarish_creature_made_of_shadows_and_teeth._1.png"),
            ("A crowded subway car, the fluorescent lights flickering and casting"
             " an eerie glow on the people inside.",
             "survey/5087_A_crowded_subway_car_the_fluorescent_lights_flickering_and_"
             "casting_an_eerie_glow_on_the_people_inside._2.png"),
            ("krita digital masterpiece of the tower of babel as a series of neural net layers",
             "survey/6532_krita_digital_masterpiece_of_the_tower_of_babel_as_a_series_of_neural_net_layers_3.png")
            
        ]
        # The survey used to record responses one at a time
        # this produced some kind of bug I don't even want to attempt
        # to hunt down where you would get more than 20 responses from a user.
        # So now we're just adding as many constraints as possible:
        # length must be 20, we add all the rating responses at once as a batch,
        # ratings must match the text of the prompt shown on the message,
        # we use a dictionary so that a repeated send just sets the same slot
        # in the survey rather than creating a duplicate. Hopefully this prevents
        # further problems.
        self.ratings = {}
        for i in range(20):
            self.ratings[self.survey_images[i][0]] = None
        self.index = 0
        
    async def rate(self, interaction, rating):
        try:
            assert interaction.message.content == self.survey_images[self.index][0]
        except AssertionError:
            await interaction.user.send("Your survey desynced, type `.signup` to try again.")
        self.ratings[interaction.message.content] = rating
        self.index += 1
        if self.index < 20:
            prompt, path = self.survey_images[self.index]
            upload = nextcord.File(path)
            await interaction.user.send(prompt,
                                        file=upload,
                                        view=self)
        else:
            self.record_survey(interaction.user.id)
            db = sqlite3.connect('db.sqlite')
            cursor = db.cursor()
            cursor.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                           (interaction.user.id, 0, 0, 0, "deprecated"))
            db.commit()
            cursor.close()
            db.close()
            await interaction.user.send("The survey is finished! You can now use the bot.")
            await interaction.user.send("**You can do so by typing `.add PROMPT`"
                                        " into the channel you first interacted with the bot in.**")
            await interaction.user.send("Please **rate images frequently**. Each "
                                        "rating is used to create models/datasets like "
                                        "LAION aesthetic "
                                        "(https://github.com/LAION-AI/laion-datasets/blob/main/laion-aesthetic.md)."
            )

    def record_survey(self, user_id):
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        for index, img in enumerate(self.survey_images):
            prompt, path = img
            assert self.ratings[prompt]
            cursor.execute("INSERT INTO survey VALUES (?,?,?)",
                           (user_id, index, self.ratings[prompt]))
        db.commit()
        cursor.close()
        db.close()
            
class AgreementSelect(nextcord.ui.View):
    def __init__(self):
        super().__init__()
        self.value = None

        
    @nextcord.ui.select(placeholder="Do you agree to these terms of service?",
                        custom_id="tosagreement",
                        options=[
                            nextcord.SelectOption(label="Yes, and understand all work with SimulacraBot is public domain",
                                                  value=True),
                            nextcord.SelectOption(label="No",
                                                  value=False),
                        ])
    async def agreement(self, select, interaction):
        if select.values[0]:
            view = RatingSurvey()
            prompt, path = view.survey_images[0]
            upload = nextcord.File(path)
            await interaction.user.send("Before you can use the bot you need to"
                                        " answer a quick 20 question rating task"
                                        " to characterize your aesthetic preferences"
                                        " and rating habits for dataset users.")
            await interaction.user.send("You may rate the images however you like"
                                        ", we try to keep it as subjective as possible."
                                        " However there are two guidelines we would "
                                        "appreciate you following:\n\n"
                                        "1. Rate images lower if they contain "
                                        "watermarks, logos, or other marks that "
                                        "intrude on the canvas. If an image has "
                                        "a disfiguring watermark across the center "
                                        "it should be no higher than a 3, if it "
                                        "includes a more subtle mark it should be "
                                        "no higher than a 6. Paintings shown on a "
                                        "wall, gallery, etc which encroaches on the "
                                        "256x256 canvas should rate no higher than a 5.\n\n"
                                        "2. As much as possible rate images on "
                                        "the merits of the image, not the prompt "
                                        "fit. Sometimes the AI produces good looking "
                                        "but poorly fitting images, these should be "
                                        "rated highly because they look good.\n\n")
            await interaction.user.send(prompt,
                                  file=upload,
                                  view=view)
    
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
        await interaction.message.author.send("You must agree to the TOS before using SimulacraBot. Type .signup")
        return
    if interaction.channel.id not in channel_whitelist:
        return
    for word in banned_words:
        if (' ' + word.lower()) in interaction.message.content.lower():
            await interaction.send("'{}' is not an allowed word by the content filter".format(word))
            return
    prompt = interaction.message.content.split(".add")[1].strip()
    if len(prompt) > 246: # Must be able to fit _grid.png on end.
        await interaction.send("The length of the prompt wouldn't fit on the "
                               "filesystem. Please shorten it and try again.")
        return
    seed = generations.get_next_seed()
    job = Job(prompt=prompt,
              cloob_checkpoint='cloob_laion_400m_vit_b_16_16_epochs',
              scale=5.,
              cutn=32,
              device='cuda:0',
              ddim_eta=0.,
              method='ddim',
              H=512,
              W=512,
              n_iter=1,
              n_samples=4,
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


    # If all images rated find images with too few ratings
    #if not to_rate:
    #    cursor.execute('''SELECT id, prompt, COUNT(rating) as num_ratings FROM 
    #                      images JOIN ratings ON images.id=ratings.iid 
    #                      WHERE images.id NOT IN 
    #                      (SELECT iid FROM ratings WHERE uid=?) 
    #                      GROUP BY images.id HAVING num_ratings < 3;''',
    #                   (interaction.message.author.id,))
    #    to_rate = cursor.fetchall()[:5]
    view = StreamRatingButtons()
    gen = view.generation
    db = sqlite3.connect('db.sqlite')
    cursor = db.cursor()
    cursor.execute("SELECT * FROM images WHERE gid=? AND idx=?",
                   (gen[0], view.index))
    image_id = cursor.fetchone()
    cursor.execute("SELECT COUNT(*) from ratings where iid=?", (image_id[0],))
    ratings = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) from flags where iid=?", (image_id[0],))
    flags = cursor.fetchone()[0]
    cursor.close()
    db.close()
    embed = nextcord.Embed(title="Feedback", description="")
    embed.add_field(name="Ratings", value=ratings)
    embed.add_field(name="Flags", value=flags)
    upload = nextcord.File(str(gen[0]) + "_" + gen[-1].replace(" ", "_").replace("/","_") +
                           "_" + str(view.index) + ".png")
        
    await interaction.channel.send(
        gen[-1],
        file=upload,
        embed=embed,
        view=view)
            
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
async def signup(interaction: nextcord.Interaction):
    if type(interaction.channel) != nextcord.channel.DMChannel:
        return

    view = AgreementSelect()
    await interaction.message.author.send("https://stability.ai/simulacrabot/terms-of-use")
    await interaction.message.author.send(
        "Before using SimulacraBot you need to agree to the terms above."
    )
    await interaction.message.author.send("These are not boilerplate, please actually "
                                         "read them. SimulacraBot is a study run by "
                                         "Stability AI LTD. and has unusual terms you "
                                         "agree to as part of participating in the study.")
    await interaction.message.author.send("A summary (which should not be construed "
                                          "as replacing the linked terms):")
    await interaction.message.author.send("1. **You agree all work you do with "
                                          "SimulacraBot, including submitted prompts, "
                                          "is public domain.** "
                                          "https://creativecommons.org/publicdomain/zero/1.0/")
    await interaction.message.author.send("2. Your work will be released pseudonymously "
                                          "as part of the Simulacra Aesthetic Captions"
                                          " dataset. You agree that the pseudonym "
                                          "method used (see terms) is not sufficiently"
                                          " identifying to qualify for protection under"
                                          " GDPR or similar laws.")
    await interaction.message.author.send("3. You will not contribute prompts to "
                                          "Simulacra Aesthetic Captions which are "
                                          "NSFW, hateful, or contain personal "
                                          "information or copyrighted material.")
    await interaction.message.author.send("4. You are responsible for your bad "
                                          "behavior, not us. Please flag bad "
                                          "behavior you see.")
    await interaction.message.author.send("Do you agree to the linked terms at "
                                          "https://stability.ai/simulacrabot/terms-of-use ? "
                                          "(they're short and readable) ",
                                          view=view)
@bot.command()
async def adduser(interaction: nextcord.Interaction):
    user = users.is_user(interaction.message.author.id)
    if (not user) or (not user[1]):
        await interaction.message.author.send("This command is restricted to admins only.")
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
        await interaction.message.author.send("This command is restricted to admins only.")
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

@bot.command()
async def stats(interaction: nextcord.Interaction):
    if type(interaction.channel) != nextcord.channel.DMChannel:
        return
    else:
        db = sqlite3.connect('db.sqlite')
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) FROM generations")
        gen_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT prompt) FROM generations")
        prompt_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM images")
        image_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ratings")
        rating_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT iid) FROM ratings")
        unique_rating_count = cursor.fetchone()[0]
        unique_rate_percent = (unique_rating_count / image_count) * 100
        
        await interaction.message.author.send(
            f"So far users have submitted {gen_count} jobs with {prompt_count} unique "
            f"prompts resulting in {image_count} images, of which {unique_rating_count} "
            f"({round(unique_rate_percent,2)}%) are rated. Users have submitted {rating_count} "
            "ratings total."
        )

@bot.command()
async def shutdown(interaction: nextcord.Interaction):
    user = users.is_user(interaction.message.author.id)
    if type(interaction.channel) != nextcord.channel.DMChannel:
        return
    if (not user) or (not user[1]):
        await interaction.message.author.send("This command is restricted to admins only.")
    bot.remove_command("add")
    @bot.command()
    async def add(interaction: nextcord.Interaction):
        await interaction.message.author.send("The bot is currently shutting down...")
    bot.remove_command("signup")
    @bot.command()
    async def signup(interaction: nextcord.Interaction):
        await interaction.message.author.send("The bot is currently shutting down...")
    await interaction.message.author.send("Commands disabled. Wait five minutes "
                                          "and then shut down the bot.")

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

