import sqlite3 as sq

database = sq.connect("Database/substrate_database")

cursor = database.cursor()



# Will interpret data as basic list.

cursor.execute('''CREATE TABLE IF NOT EXISTS substrate (
               
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               Material TEXT,
               Shape TEXT,
               Size_Width REAL,
               Size_Height REAL,
               Color REAL,
               Position_X REAL,
               Position_Y REAL

               )'''
    )


cursor.execute("INSERT INTO substrate (Material) VALUES (?)", ("Graphene",))


cursor.execute("SELECT * FROM substrate")
rows = cursor.fetchall()

# Print each row
for row in rows:
    print(row)
