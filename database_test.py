import sqlite3 as sq
import csv
from test import *

database = sq.connect("Database/substrate_database")

cursor = database.cursor()



# Will interpret data as basic list.

cursor.execute('''
    CREATE TABLE IF NOT EXISTS substrate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        idx INTEGER,
        area REAL,
        perimeter REAL,
        aspect_ratio REAL,
        solidity REAL,
        circularity REAL,
        centroid_x REAL,
        centroid_y REAL,
        bbox_x INTEGER,
        bbox_y INTEGER,
        bbox_w INTEGER,
        bbox_h INTEGER,
        entropy REAL
    )
''')



for entry in shape_features:

  
    cursor.execute(
        '''
        INSERT INTO substrate (
            idx, area, perimeter, aspect_ratio, solidity, circularity,
            centroid_x, centroid_y, bbox_x, bbox_y, bbox_w, bbox_h, entropy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            entry["idx"],
            entry["area"],
            entry["perimeter"],
            entry["aspect_ratio"],
            entry["solidity"],
            entry["circularity"],
            entry["centroid_x"],
            entry["centroid_y"],
            entry["bbox_x"],
            entry["bbox_y"],
            entry["bbox_w"],
            entry["bbox_h"],
            entry["entropy"],
        )
    )
    


cursor.execute("SELECT * FROM substrate")
rows = cursor.fetchall()

header = []

for head in cursor.description:
    print(head)
    header.append(head[0])

with open('substate.csv', 'w', newline='') as f:

    write = csv.writer(f)

    write.writerow(header)

    write.writerows(rows)

# Print each row
# for row in rows:
#     print(row)
# 
