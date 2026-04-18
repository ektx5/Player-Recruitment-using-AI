import pandas as pd



def load_data():
    players = pd.read_csv("data/players.csv")
    appearances = pd.read_csv("data/appearances.csv")
    valuations = pd.read_csv("data/player_valuations.csv")
    return players, appearances, valuations

def merge_data(players, appearances, valuations):

    # -------------------------------
    # Aggregate stats
    # -------------------------------
    stats = appearances.groupby("player_id").agg({
        "minutes_played": "sum",
        "goals": "sum",
        "assists": "sum"
    }).reset_index()

    # -------------------------------
    # Get latest valuation + club + league
    # -------------------------------
    valuations = valuations.sort_values("date", ascending=False)
    valuations = valuations.drop_duplicates(subset=["player_id"])

    valuations = valuations[[
        "player_id",
        "market_value_in_eur",
        "current_club_name",
        "player_club_domestic_competition_id"
    ]]

    # -------------------------------
    # Merge
    # -------------------------------
    df = players.merge(stats, on="player_id", how="inner")
    df = df.merge(valuations, on="player_id", how="inner")

    # -------------------------------
    # Basic fields
    # -------------------------------
    df["player_name"] = df["name"]

    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["age"] = 2024 - df["date_of_birth"].dt.year

    # -------------------------------
    # Nationality
    # -------------------------------
    if "country_of_citizenship" in df.columns:
        df["nationality"] = df["country_of_citizenship"]
    elif "country_of_birth" in df.columns:
        df["nationality"] = df["country_of_birth"]
    else:
        df["nationality"] = "Unknown"

    # -------------------------------
    # Position safe
    # -------------------------------
    if "position" not in df.columns:
        df["position"] = "Unknown"

    # -------------------------------
    # Select columns
    # -------------------------------
    df = df[[
        "player_name",
        "age",
        "nationality",
        "position",
        "current_club_name",
        "player_club_domestic_competition_id",
        "minutes_played",
        "goals",
        "assists",
        "market_value_in_eur"
    ]]

    # -------------------------------
    # Rename columns
    # -------------------------------
    df = df.rename(columns={
        "market_value_in_eur": "market_value",
        "player_club_domestic_competition_id": "league"
    })

    # -------------------------------
    # Clean
    # -------------------------------
    df = df.dropna()
    df = df[df["minutes_played"] > 300]
    df = df[df["age"] < 40]

    return df

def feature_engineering(df):
    df["goals_per90"] = df["goals"] / df["minutes_played"] * 90
    df["assists_per90"] = df["assists"] / df["minutes_played"] * 90
    return df