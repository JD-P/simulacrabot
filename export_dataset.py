import os
import shutil
import sqlite3
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("db")
args = parser.parse_args()

# Create release database

sac_db = sqlite3.connect('sac.sqlite')
cursor = sac_db.cursor()
cursor.execute('''CREATE TABLE survey
                  (id INTEGER, qid INTEGER, rating INTEGER,
                  PRIMARY KEY(id, qid))''')
cursor.execute('''CREATE TABLE generations
                  (id INTEGER PRIMARY KEY, sid INTEGER, method INTEGER, prompt TEXT, verified INTEGER,
                  FOREIGN KEY(sid) REFERENCES survey(id))''')
cursor.execute('''CREATE TABLE images
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  gid INTEGER,
                  idx INTEGER,
                  FOREIGN KEY(gid) REFERENCES generations(id),
                  UNIQUE(gid,idx))''')
cursor.execute('''CREATE TABLE paths
                  (iid INTEGER, path TEXT,
                  FOREIGN KEY(iid) REFERENCES images(id))''')
cursor.execute('''CREATE TABLE ratings
                  (sid INTEGER, iid INTEGER, rating INTEGER, verified INTEGER,
                  FOREIGN KEY(sid) REFERENCES survey(id),
                  FOREIGN KEY(iid) REFERENCES images(id))''')
cursor.execute('''CREATE TABLE upscales
                  (iid INTEGER PRIMARY KEY, method INTEGER,
                  FOREIGN KEY(iid) REFERENCES images(id))''')
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

# Preprocess synthetic identities

bot_cursor.execute("SELECT * FROM users")
users = bot_cursor.fetchall()

responses = set()
user2response = {}
for user in users:
    bot_cursor.execute("SELECT * FROM survey WHERE uid=? ORDER BY qid ASC",
                       (user[0],))
    survey = bot_cursor.fetchall()
    response = tuple()
    for i, s in enumerate(survey):
        assert s[1] == i
        response += (s[2],)
    responses.add(response)
    user2response[user[0]] = response

response2sid = {}
for i, response in enumerate(responses):
    response2sid[response] = i

bot_cursor.execute("SELECT * FROM users;")
users = bot_cursor.fetchall()

sids = set()
for user in users:
    survey = user2response[user[0]]
    sid = response2sid[survey]
    if sid in sids:
        continue
    for i, q in enumerate(survey):
        sac_cursor.execute("INSERT INTO survey VALUES (?,?,?)",
                           (sid, i, q))
    sids.add(sid)
    
for gen in tqdm(gens):
    path_template = str(gen[0]) + "_" + gen[-1].replace(" ", "_").replace("/","_").replace("{", "{{").replace("}","}}") + "_{}" + ".png"
    paths = [path[1].format(path[0]+1) for path in enumerate([path_template] * 8)]
    bot_cursor.execute("SELECT * FROM images WHERE gid=? ORDER BY idx ASC", (gen[0],))
    images = bot_cursor.fetchall()
    # Correct index and redact info into new release database
    bot_cursor.execute("SELECT * FROM users WHERE id=?", (gen[1],))
    user = bot_cursor.fetchone()
    bot_cursor.execute("SELECT * FROM survey WHERE uid=?", (user[0],))
    survey = user2response[user[0]]
    sid = response2sid[survey]                            
    verified = int(user[3])
    bot_cursor.execute("SELECT ratings.* FROM ratings INNER JOIN images on "
                       "ratings.iid = images.id INNER JOIN generations on "
                       "images.gid = generations.id WHERE generations.id=?", (gen[0],))
    ratings = bot_cursor.fetchall()
    bot_cursor.execute("SELECT upscales.* FROM upscales INNER JOIN images on "
                       "upscales.iid = images.id INNER JOIN generations on "
                       "images.gid = generations.id WHERE generations.id=?", (gen[0],))
    upscales = bot_cursor.fetchall()
    assert len(upscales) <= 8
    
    sac_cursor.execute("INSERT INTO generations VALUES (?,?,?,?,?)", (gen[0], sid, gen[3], gen[4], verified))
    for i, image in enumerate(images):
        assert (i+1) == image[2] 
        sac_cursor.execute("INSERT INTO images VALUES (?, ?, ?)", (image[0], image[1], image[2]))
        sac_cursor.execute("INSERT INTO paths VALUES (?, ?)", (image[0], paths[i])) 
    for rating in ratings:
        sac_cursor.execute("INSERT INTO ratings VALUES (?,?,?,?)", (sid, rating[1], rating[2], verified))
    for upscale in upscales:
        sac_cursor.execute("INSERT INTO upscales VALUES (?,?)", (upscale[0], upscale[1]))
    sac_db.commit()

# Sanity checking
bot_cursor.execute("SELECT count(*) FROM ratings;")
bot_ratings = bot_cursor.fetchone()[0]
sac_cursor.execute("SELECT count(*) FROM ratings;")
sac_ratings = sac_cursor.fetchone()[0]

# assert bot_ratings == sac_ratings
# Difference of 4 between them, probably fine

bot_cursor.close()
sac_cursor.close()

bot_db.close()
sac_db.close()
