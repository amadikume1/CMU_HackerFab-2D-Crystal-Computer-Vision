import sqlite3 as sq
import csv
database = sq.connect("Database/substrate_database")

cursor = database.cursor()



# Will interpret data as basic list.

information = [["Wafer1", "Graphene", "Triangle", 10, 10, "GREEN", 150, 180],
               ["Wafer2", "Graphene", "Square", 15, 10, "GREEN", 150, 180]]

cursor.execute('''CREATE TABLE IF NOT EXISTS substrate (
               
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               Wafer_ID TEXT,
               Material TEXT,
               Shape TEXT,
               Size_Width REAL,
               Size_Height REAL,
               Color TEXT,
               Position_X REAL,
               Position_Y REAL

               )'''
    )



for entry in information:
    cursor.execute(f"INSERT INTO substrate (Wafer_ID, Material, Shape, Size_Width, Size_Height, Color, Position_X, Position_Y) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (entry[0],entry[1], entry[2], entry[3], entry[4],entry[5], entry[6], entry[7],))


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
for row in rows:
    print(row)
