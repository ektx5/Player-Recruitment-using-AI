from src.preprocessing import load_data, merge_data, feature_engineering
from src.clustering import train_clustering
from src.prediction import train_price_model
import os

# Create data and models dirs if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
players, appearances, valuations = load_data()

# Process
df = merge_data(players, appearances, valuations)
df = feature_engineering(df)

# Train clustering (generates role_cluster) and save clustering model
df = train_clustering(df)

# Train price model and save
train_price_model(df)

# Save
df.to_csv("data/final_dataset.csv", index=False)

print("✅ Dataset and models created successfully!")