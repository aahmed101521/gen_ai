import sqlite3

# Create a local database file
conn = sqlite3.connect('electronics_store.db')
cursor = conn.cursor()

print("1. Forging the Database...")
# Table 1: Inventory
cursor.execute('''
CREATE TABLE IF NOT EXISTS inventory (
    product_id INTEGER PRIMARY KEY,
    brand TEXT,
    category TEXT,
    color TEXT,
    price REAL,
    stock_quantity INTEGER
)
''')

# Table 2: Discounts
cursor.execute('''
CREATE TABLE IF NOT EXISTS discounts (
    discount_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    discount_percent REAL,
    FOREIGN KEY(product_id) REFERENCES inventory(product_id)
)
''')

print("2. Stocking the Shelves...")
# Insert sample electronics
products = [
    (1, 'Apple', 'Smartphone', 'White', 999.00, 45),
    (2, 'Apple', 'Smartphone', 'Black', 999.00, 30),
    (3, 'Samsung', 'Smartphone', 'Black', 850.00, 60),
    (4, 'Sony', 'Headphones', 'Silver', 350.00, 15),
    (5, 'Dell', 'Laptop', 'Silver', 1200.00, 10),
    (6, 'Apple', 'Laptop', 'Space Gray', 1500.00, 25)
]
cursor.executemany('INSERT OR IGNORE INTO inventory VALUES (?,?,?,?,?,?)', products)

# Add a discount (20% off the Sony Headphones)
cursor.execute('INSERT OR IGNORE INTO discounts VALUES (1, 4, 20.0)')

conn.commit()
conn.close()

print("✅ Success! 'electronics_store.db' is ready for the AI.")