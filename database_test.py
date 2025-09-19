import sqlite3 as sq
import csv

database = sq.connect("Database/substrate_database")
cursor = database.cursor()

# Example data as lists
information = [
    ["Wafer1", "Graphene", "Triangle", 10, 10, "GREEN", 150, 180],
    ["Wafer2", "Graphene", "Square", 15, 10, "GREEN", 150, 180]
]

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS substrate (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Wafer_ID TEXT,
        Material TEXT,
        Shape TEXT,
        Size_Width REAL,
        Size_Height REAL,
        Color TEXT,
        Position_X REAL,
        Position_Y REAL
    )
''')

# Insert the data from lists
for entry in information:
    cursor.execute('''
        INSERT INTO substrate (
            Wafer_ID, Material, Shape, Size_Width, Size_Height, Color, Position_X, Position_Y
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', entry)

database.commit()  # Make sure to commit the changes

# Fetch all rows
cursor.execute("SELECT * FROM substrate")
rows = cursor.fetchall()

# Get headers
header = [desc[0] for desc in cursor.description]

# Save to CSV
with open('substrate.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

# Optionally, print each row
for row in rows:
    print(row)

database.close()
