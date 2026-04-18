from src.preprocessing import load_data, merge_data, feature_engineering

# Load data
players, appearances, valuations = load_data()

# Process
df = merge_data(players, appearances, valuations)
df = feature_engineering(df)

# Save
df.to_csv("data/final_dataset.csv", index=False)

print("✅ Dataset created successfully!")