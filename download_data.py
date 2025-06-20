import kagglehub
import os

# Define the directory to save the data
data_dir = "data/upi_transactions"
os.makedirs(data_dir, exist_ok=True)

# Download latest version of the dataset
print("Downloading UPI Payment Transactions dataset from Kaggle...")
path = kagglehub.dataset_download(
    "devildyno/upi-payment-transactions-dataset",
    path=data_dir
)

print(f"Dataset downloaded successfully to: {path}") 