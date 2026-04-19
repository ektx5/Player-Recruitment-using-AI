import pandas as pd
from datetime import datetime



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
    # Drop conflicting columns from players
    # (real Transfermarkt data has these in both tables)
    # -------------------------------
    drop_cols = [c for c in ["current_club_name", "player_club_domestic_competition_id",
                              "market_value_in_eur", "current_club_id"] if c in players.columns]
    players_clean = players.drop(columns=drop_cols, errors="ignore")

    # -------------------------------
    # Merge
    # -------------------------------
    df = players_clean.merge(stats, on="player_id", how="inner")
    df = df.merge(valuations, on="player_id", how="inner")

    # -------------------------------
    # Basic fields
    # -------------------------------
    df["player_name"] = df["name"]

    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    current_year = datetime.now().year
    df["age"] = current_year - df["date_of_birth"].dt.year

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
    # Position safe + normalize names
    # -------------------------------
    if "position" not in df.columns:
        df["position"] = "Unknown"

    # Transfermarkt uses "Attack" / "Midfield" — normalize
    position_map = {
        "Attack": "Forward",
        "Midfield": "Midfielder",
    }
    df["position"] = df["position"].replace(position_map)

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
    # Rename columns and Map League Names
    # -------------------------------
    df = df.rename(columns={
        "market_value_in_eur": "market_value",
        "player_club_domestic_competition_id": "league"
    })

    league_map = {
        "GB1": "Premier League",
        "ES1": "La Liga",
        "IT1": "Serie A",
        "L1": "Bundesliga",
        "FR1": "Ligue 1",
        "PO1": "Liga Portugal",
        "NL1": "Eredivisie",
        "TR1": "Süper Lig",
        "GR1": "Super League Greece",
        "RU1": "Russian Premier League",
        "BE1": "Jupiler Pro League",
        "DK1": "Danish Superliga",
        "SC1": "Scottish Premiership",
        "UKR1": "Ukrainian Premier League",
        "BRA1": "Campeonato Brasileiro",
        "MLS1": "Major League Soccer",
        "A1": "Austrian Bundesliga",
        "ARG1": "Liga Profesional (Argentina)",
        "AUS1": "A-League Men",
        "C1": "Swiss Super League",
        "COL1": "Categoría Primera A",
        "JAP1": "J1 League",
        "KR1": "SuperSport HNL (Croatia)",
        "MEX1": "Liga MX",
        "NO1": "Eliteserien",
        "PL1": "Ekstraklasa",
        "RO1": "Romanian SuperLiga",
        "RSK1": "K League 1",
        "SA1": "Saudi Pro League",
        "SE1": "Allsvenskan",
        "SER1": "Serbian SuperLiga",
        "TS1": "Czech First League"
    }
    df["league"] = df["league"].replace(league_map)

    # -------------------------------
    # Clean
    # -------------------------------
    df = df.dropna()
    df = df[df["minutes_played"] > 0] # Need > 0 to avoid division by zero in per90 stats

    return df

def feature_engineering(df):
    df["goals_per90"] = df["goals"] / df["minutes_played"] * 90
    df["assists_per90"] = df["assists"] / df["minutes_played"] * 90
    return df