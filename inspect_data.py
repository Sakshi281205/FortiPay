import pandas as pd
import io

# Load the dataset
df = pd.read_csv('transactions.csv')

# Print the first 5 rows
print("First 5 rows of transactions.csv:")
print(df.head(5))

# Get column info into a string buffer
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

# Print column names and data types
print("\nColumn Info:")
print(info_str) 