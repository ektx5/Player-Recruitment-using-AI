import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Player Recruitment AI", layout="wide")

# Styling
st.markdown("""
<style>
body {background-color: #0e1117;}
h1, h2, h3 {color: #00ff88;}
</style>
""", unsafe_allow_html=True)

# Load
df = pd.read_csv("data/final_dataset.csv")
model = joblib.load("models/price_model.pkl")

st.title("⚽ Player Recruitment Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

age_range = st.sidebar.slider("Age", 16, 40, (18, 30))

positions = st.sidebar.multiselect(
    "Position",
    df["position"].unique(),
    default=list(df["position"].unique())
)

filtered_df = df[
    (df["age"] >= age_range[0]) &
    (df["age"] <= age_range[1]) &
    (df["position"].isin(positions))
]

# Table
st.subheader("Scouting Pool")
st.dataframe(filtered_df.head(100))

# Select player
player = st.selectbox("Select Player", filtered_df["player_name"])

player_data = filtered_df[filtered_df["player_name"] == player].iloc[0]

# Info
st.subheader(player)

st.write(f"🌍 {player_data['nationality']}")
st.write(f"🏟 {player_data['current_club_name']}")
st.write(f"🏆 {player_data['league']}")
st.write(f"📍 {player_data['position']}")

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Age", int(player_data["age"]))
col2.metric("Goals/90", round(player_data["goals_per90"], 2))
col3.metric("Assists/90", round(player_data["assists_per90"], 2))

# Prediction
features = [[
    player_data["age"],
    player_data["goals_per90"],
    player_data["assists_per90"],
    player_data["role_cluster"]
]]

predicted = model.predict(features)[0]

st.subheader("Market Value")

col4, col5 = st.columns(2)

col4.metric("Actual", int(player_data["market_value"]))
col5.metric("Predicted", int(predicted))

# Undervalued logic
if predicted > player_data["market_value"]:
    st.success("Undervalued player")
else:
    st.error("Overvalued player")

# Similar players
st.subheader("Similar Players")

sim = cosine_similarity(df[["goals_per90", "assists_per90"]])

idx = df[df["player_name"] == player].index[0]

scores = list(enumerate(sim[idx]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)

top = [df.iloc[i[0]]["player_name"] for i in scores[1:6]]

st.write(top)