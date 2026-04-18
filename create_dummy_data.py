import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. players.csv
players_data = {
    "player_id": range(1, 101),
    "name": [f"Player {i}" for i in range(1, 101)],
    "date_of_birth": [(datetime(1995, 1, 1) + timedelta(days=np.random.randint(0, 3650))).strftime("%Y-%m-%d") for _ in range(100)],
    "country_of_citizenship": np.random.choice(["Spain", "France", "Germany", "Brazil", "Argentina", "England"], 100),
    "position": np.random.choice(["Forward", "Midfielder", "Defender", "Goalkeeper"], 100)
}
pd.DataFrame(players_data).to_csv("data/players.csv", index=False)

# 2. appearances.csv
appearances_data = {
    "player_id": np.random.choice(range(1, 101), 500),
    "minutes_played": np.random.randint(10, 90, 500),
    "goals": np.random.choice([0, 1, 2], 500, p=[0.8, 0.15, 0.05]),
    "assists": np.random.choice([0, 1, 2], 500, p=[0.85, 0.10, 0.05]),
}
pd.DataFrame(appearances_data).to_csv("data/appearances.csv", index=False)

# 3. player_valuations.csv
valuations_data = {
    "player_id": range(1, 101),
    "date": [(datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d") for _ in range(100)],
    "market_value_in_eur": np.random.randint(500000, 50000000, 100),
    "current_club_name": np.random.choice(["Real Madrid", "FC Barcelona", "Bayern Munich", "PSG", "Man City", "Arsenal"], 100),
    "player_club_domestic_competition_id": np.random.choice(["LaLiga", "Ligue 1", "Bundesliga", "Premier League"], 100)
}
pd.DataFrame(valuations_data).to_csv("data/player_valuations.csv", index=False)

print("Dummy data generated.")
