import sqlite3 as sq
import csv

# Connect to your existing database
database = sq.connect("Database/substrate_sobel_new.db")
cursor = database.cursor()

# Read entire table INCLUDING id, Wafer_ID, Material, Shape, etc.
cursor.execute("SELECT * FROM substrate_sobel_new")
rows = cursor.fetchall()

# Extract column names (these will be exactly the 9 columns you listed)
header = [col[0] for col in cursor.description]

# Export to CSV
with open("Database/substrate_sobel_new.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)   # writes the columns
    writer.writerows(rows)    # writes the data

# Print rows for confirmation
for row in rows:
    print(row)
