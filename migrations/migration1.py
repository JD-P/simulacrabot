import os
import sqlite3
from tqdm import tqdm


if not os.path.exists("migration.sqlite"):
    migrate_db = sqlite3.connect('migration.sqlite')
    migrate_cursor = migrate_db.cursor()
    migrate_cursor.execute('''CREATE TABLE users
                           (id INTEGER PRIMARY KEY, admin INTEGER, banned INTEGER,
                           verified INTEGER, name TEXT)''')
    migrate_cursor.execute('''CREATE TABLE generations
                           (id INTEGER PRIMARY KEY, uid INTEGER, mid INTEGER, 
                           method INTEGER, prompt TEXT, 
                           FOREIGN KEY(uid) REFERENCES users(id))''')
    migrate_cursor.execute('''CREATE TABLE images
                           (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           gid INTEGER,
                           idx INTEGER,
                           FOREIGN KEY(gid) REFERENCES generations(id),
                           UNIQUE(gid,idx))''')
    migrate_cursor.execute('''CREATE TABLE ratings
                           (uid INTEGER, iid INTEGER, rating INTEGER,
                           FOREIGN KEY(uid) REFERENCES users(id),
                           FOREIGN KEY(iid) REFERENCES images(id),
                           PRIMARY KEY(uid, iid))''')
    migrate_cursor.execute('''CREATE TABLE flags
                           (uid INTEGER, iid INTEGER, 
                           FOREIGN KEY(uid) REFERENCES users(id),
                           FOREIGN KEY(iid) REFERENCES images(id),
                           PRIMARY KEY(uid, iid))''')
    migrate_cursor.execute('''CREATE TABLE upscales
                           (iid INTEGER, method INTEGER,
                           FOREIGN KEY(iid) REFERENCES images(id),
                           PRIMARY KEY(iid))''')

    current_db = sqlite3.connect('db.sqlite')
    current_cursor = current_db.cursor()

    current_cursor.execute("SELECT * FROM users")
    users = current_cursor.fetchall()
    for user in users:
        migrate_cursor.execute("INSERT INTO users VALUES (?,?,?,?,?)", (user[0],
                                                                        user[1],
                                                                        user[2],
                                                                        user[3],
                                                                        user[4]))
    
    current_cursor.execute("SELECT * FROM generations")
    gens = current_cursor.fetchall()
    for gen in tqdm(gens):
        migrate_cursor.execute("INSERT INTO generations VALUES (?,?,?,?,?)", (gen[0],
                                                                              gen[1],
                                                                              gen[2],
                                                                              gen[3],
                                                                              gen[4]))
        for i in range(1,9):
            migrate_cursor.execute("INSERT INTO images(gid,idx) VALUES (?,?)", (gen[0],
                                                                       i))
        
        migrate_cursor.execute("SELECT * FROM images WHERE gid=? AND idx=1",
                               (gen[0],))
        zero_id = migrate_cursor.fetchone()[0]
        current_cursor.execute("SELECT * FROM ratings WHERE gid=?", (gen[0],))
        ratings = current_cursor.fetchall()
        for rating in ratings:
            migrate_cursor.execute("INSERT INTO ratings VALUES (?,?,?)",
                                   (rating[0], zero_id, rating[2]))
        current_cursor.execute("SELECT * FROM flags WHERE gid=?", (gen[0],))
        flags = current_cursor.fetchall()
        for flag in flags:
            migrate_cursor.execute("SELECT id FROM images WHERE gid=?",
                                   (flag[1],))
            image_ids = [x[0] for x in migrate_cursor.fetchall()]
            for iid in image_ids:
                migrate_cursor.execute("INSERT INTO flags VALUES (?,?)",
                                       (flag[0], iid))
        current_cursor.execute("SELECT * FROM upscales WHERE gid=?", (gen[0],))
        upscales = current_cursor.fetchall()
        for upscale in upscales:
            migrate_cursor.execute("SELECT id FROM images WHERE gid=? AND idx=?",
                                              (upscale[0], upscale[1]))
            image_id = migrate_cursor.fetchone()[0]
            assert image_id
            migrate_cursor.execute("INSERT INTO upscales VALUES (?,?)",
                                   (image_id, upscale[2]))
    migrate_db.commit()
    current_cursor.close()
    migrate_cursor.close()
    current_db.close()
    migrate_db.close()
