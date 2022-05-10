import os
import shutil
import sqlite3
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("db")
args = parser.parse_args()

# Prepare dataset folder

try:
    os.mkdir("dataset")
except FileExistsError:
    pass

# Create release database

sac_db = sqlite3.connect('dataset/sac.sqlite')
cursor = sac_db.cursor()
cursor.execute('''CREATE TABLE generations
                  (id INTEGER PRIMARY KEY, method INTEGER, prompt TEXT, verified INTEGER)''')
cursor.execute('''CREATE TABLE ratings
                  (gid INTEGER, rating INTEGER,
                  FOREIGN KEY(gid) REFERENCES generations(id))''')
cursor.execute('''CREATE TABLE upscales
                  (gid INTEGER, choice INTEGER, method INTEGER,
                  FOREIGN KEY(gid) REFERENCES generations(id),
                  PRIMARY KEY(gid, choice))''')
sac_db.commit()
cursor.close()


# Retrieve generations from current database

bot_db = sqlite3.connect(args.db)
cursor = bot_db.cursor()
cursor.execute("select * from generations;")
gens = cursor.fetchall()
cursor.close()

bot_cursor = bot_db.cursor()
sac_cursor = sac_db.cursor()
for i, gen in enumerate(tqdm(gens)):
    path_template = str(gen[0]) + "_" + gen[-1].replace(" ", "_").replace("/","_") + "_{}" + ".png"
    paths = [path[1].format(path[0]+1) for path in enumerate([path_template] * 8)]
    for path in paths:
        shutil.copy(path, "dataset/{}".format(path))
    # Correct index and redact info into new release database
    bot_cursor.execute("SELECT * FROM users WHERE id=?", (gen[1],))
    user = bot_cursor.fetchone()
    verified = int(user[3])
    bot_cursor.execute("SELECT * FROM ratings WHERE gid=?", (gen[0],))
    ratings = bot_cursor.fetchall()
    bot_cursor.execute("SELECT * FROM upscales WHERE gid=?", (gen[0],))
    upscales = bot_cursor.fetchall()
    sac_cursor.execute("INSERT INTO generations VALUES (?,?,?,?)", (i, gen[3], gen[4], verified))
    for rating in ratings:
        sac_cursor.execute("INSERT INTO ratings VALUES (?,?)", (i, rating[2]))
    for upscale in upscales:
        sac_cursor.execute("INSERT INTO upscales VALUES (?,?,?)", (i, upscale[1], upscale[2]))
    sac_db.commit()

bot_cursor.close()
sac_cursor.close()

bot_db.close()
sac_db.close()
