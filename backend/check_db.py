import sqlite3
conn = sqlite3.connect("training_data.db")
c = conn.cursor()
c.execute("SELECT id, spoken_text FROM training_items LIMIT 5")
for row in c.fetchall():
    print(row)
conn.close()
