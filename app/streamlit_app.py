import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Player Recruitment AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# LIGHT / WHITE CSS THEME
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #f8fafc; /* Very light gray/white background */
    --bg-card: #ffffff;    /* Pure white cards */
    --border-glass: #e2e8f0; /* Light gray borders */
    --border-glow: #3b82f6;  /* Blue focus ring */
    
    --accent-primary: #2563eb; /* Primary blue */
    --accent-secondary: #0ea5e9; /* Light blue */
    --accent-green: #10b981; /* Emerald green for positive */
    --accent-red: #ef4444; /* Red for negative */
    --accent-gold: #f59e0b; /* Gold/Amber */
    --accent-purple: #8b5cf6; /* Purple */
    
    --text-primary: #0f172a; /* Dark text */
    --text-secondary: #475569; /* Medium gray text */
    --text-muted: #64748b; /* Lighter gray text */
    
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border-glass) !important;
}

/* Hide streamlit header */
header[data-testid="stHeader"] { background: transparent !important; }

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #cbd5e1 !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--accent-primary) !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid var(--border-glass) !important;
    box-shadow: var(--shadow-sm);
    background: white;
}

/* Section titles */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--accent-primary);
    display: inline-block;
}

/* Hero */
.hero-header {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 1.5rem;
    background: white;
    border-radius: 16px;
    border: 1px solid var(--border-glass);
    box-shadow: var(--shadow-sm);
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #1e40af, #2563eb, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 500;
    margin-top: 8px;
}

/* White container */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}
.glass-card:hover {
    box-shadow: var(--shadow-md);
    border-color: #cbd5e1;
}

/* Player name */
.pname {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}

/* Position badge */
.pos-badge {
    display: inline-block;
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Info row */
.info-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid var(--border-glass);
    margin-bottom: 8px;
}
.info-icon { font-size: 1.2rem; width: 24px; text-align: center; }
.info-lbl { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; }
.info-val { font-size: 0.95rem; color: var(--text-primary); font-weight: 600; }

/* Stat pill */
.stat-pill {
    background: white;
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    flex: 1;
    box-shadow: var(--shadow-sm);
}
.stat-pill-lbl { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; margin-bottom: 4px; }
.stat-pill-val { font-size: 1.2rem; font-weight: 800; color: var(--accent-primary); white-space: nowrap; }

/* Value card */
.val-card {
    background: white;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    border: 1px solid var(--border-glass);
    box-shadow: var(--shadow-sm);
}
.val-lbl { font-size: 0.65rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.val-amt { font-size: 1.1rem; font-weight: 800; color: var(--text-primary); white-space: nowrap; }
.val-cyan { border-top: 4px solid var(--accent-primary); }
.val-cyan .val-lbl { color: var(--accent-primary); }
.val-purple { border-top: 4px solid var(--accent-purple); }
.val-purple .val-lbl { color: var(--accent-purple); }

/* Badge */
.badge-uv {
    display: inline-block;
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
    padding: 6px 18px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 700;
}
.badge-ov {
    display: inline-block;
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 6px 18px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 700;
}

/* Similar card */
.sim-card {
    background: white;
    border: 1px solid var(--border-glass);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}
.sim-card:hover {
    border-color: var(--accent-primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.sim-rank { font-size: 0.65rem; font-weight: 800; color: var(--accent-gold); text-transform: uppercase; letter-spacing: 0.08em; }
.sim-name { font-size: 0.95rem; font-weight: 800; color: var(--text-primary); margin: 4px 0; }
.sim-meta { font-size: 0.75rem; color: var(--text-secondary); font-weight: 500; }
.sim-score { font-size: 0.75rem; color: var(--accent-primary); font-weight: 700; margin-top: 8px; background: #eff6ff; padding: 4px; border-radius: 6px;}

/* Divider */
.divider { border: none; height: 1px; background: var(--border-glass); margin: 2rem 0; }

/* Sidebar heading */
.sidebar-head {
    color: var(--text-primary);
    font-weight: 800;
    font-size: 1.25rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def fmt_money(val):
    return f"€{int(val):,}"

@st.cache_data
def load_dataset():
    return pd.read_csv("data/final_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/price_model.pkl")

df = load_dataset()
model = load_model()


# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-header">'
    '<div class="hero-title">⚽ Player Recruitment AI</div>'
    '<div class="hero-subtitle">Intelligent scouting powered by machine learning — Discover, Compare &amp; Recruit</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-head">🎯 Scout Filters</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Only set sliders bounds if there is data
    min_age = int(df["age"].min()) if not df.empty else 16
    max_age = int(df["age"].max()) if not df.empty else 40
    if min_age == max_age: max_age += 1 # safe fallback
    
    age_range = st.slider("📅 Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    st.markdown("*(Leave filters blank to include ALL)*")
    positions = st.multiselect("📍 Positions", sorted(df["position"].unique()))
    leagues = st.multiselect("🏆 Leagues", sorted(df["league"].unique()))
    
    # Dynamically filter available clubs based on selected leagues
    if leagues:
        available_clubs = df[df["league"].isin(leagues)]["current_club_name"].unique()
    else:
        available_clubs = df["current_club_name"].unique()
        
    clubs = st.multiselect("🏟️ Clubs", sorted(available_clubs))

    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn")


# ─────────────────────────────────────────────────────────
# APPLY FILTERS LOGIC
# ─────────────────────────────────────────────────────────
filtered_df = df.copy()

if age_range:
    filtered_df = filtered_df[(filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1])]

if positions:
    filtered_df = filtered_df[filtered_df["position"].isin(positions)]

if leagues:
    filtered_df = filtered_df[filtered_df["league"].isin(leagues)]

if clubs:
    filtered_df = filtered_df[filtered_df["current_club_name"].isin(clubs)]


# ─────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Players", f"{len(filtered_df):,}")
k2.metric("Avg Age", round(filtered_df["age"].mean(), 1) if len(filtered_df) else "—")
k3.metric("Avg Goals/90", round(filtered_df["goals_per90"].mean(), 2) if len(filtered_df) else "—")
k4.metric("Avg Assists/90", round(filtered_df["assists_per90"].mean(), 2) if len(filtered_df) else "—")

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SCOUTING POOL TABLE
# ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Scouting Pool</div>', unsafe_allow_html=True)

display_cols = [
    "player_name", "age", "nationality", "position",
    "current_club_name", "league", "goals_per90",
    "assists_per90", "market_value",
]

st.dataframe(
    filtered_df[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=400,
    column_config={
        "player_name": st.column_config.TextColumn("Player", width="medium"),
        "age": st.column_config.NumberColumn("Age", format="%d"),
        "nationality": st.column_config.TextColumn("Nationality"),
        "position": st.column_config.TextColumn("Position"),
        "current_club_name": st.column_config.TextColumn("Club", width="medium"),
        "league": st.column_config.TextColumn("League"),
        "goals_per90": st.column_config.NumberColumn("G/90", format="%.2f"),
        "assists_per90": st.column_config.NumberColumn("A/90", format="%.2f"),
        "market_value": st.column_config.NumberColumn("Market Value (€)", format="€%,d"),
    },
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# PLAYER SELECTION
# ─────────────────────────────────────────────────────────
if len(filtered_df) == 0:
    st.warning("No players match your filters. Adjust the sidebar criteria.")
    st.stop()

st.markdown('<div class="section-title">🔍 Player Deep Dive</div>', unsafe_allow_html=True)

player_list = filtered_df["player_name"].tolist()
player = st.selectbox("Select a player to analyse", player_list)

player_data = filtered_df[filtered_df["player_name"] == player].iloc[0]


# ─────────────────────────────────────────────────────────
# PLAYER PROFILE + MARKET VALUE  (side by side)
# ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

# ── LEFT: Player Profile Card ──
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pname">{player_data["player_name"]}</div>'
        f'<div class="pos-badge">{player_data["position"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    info_items = [
        ("🌍", "Nationality", player_data["nationality"]),
        ("🏟️", "Club", player_data["current_club_name"]),
        ("🏆", "League", player_data["league"]),
        ("📅", "Age", f'{int(player_data["age"])} years'),
    ]
    for icon, label, value in info_items:
        st.markdown(
            f'<div class="info-row">'
            f'<span class="info-icon">{icon}</span>'
            f'<div><div class="info-lbl">{label}</div><div class="info-val">{value}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    s1, s2, s3, s4 = st.columns(4)
    for col, (lbl, val) in zip(
        [s1, s2, s3, s4],
        [
            ("Minutes", f"{int(player_data['minutes_played']):,}"),
            ("Goals / 90", f"{player_data['goals_per90']:.2f}"),
            ("Assists / 90", f"{player_data['assists_per90']:.2f}"),
            ("Role Cluster", str(int(player_data["role_cluster"]))),
        ],
    ):
        col.markdown(
            f'<div class="stat-pill">'
            f'<div class="stat-pill-lbl">{lbl}</div>'
            f'<div class="stat-pill-val">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ── RIGHT: Market Valuation Card ──
with right_col:
    features = [[
        player_data["age"],
        player_data["goals_per90"],
        player_data["assists_per90"],
        player_data["role_cluster"],
    ]]
    predicted = model.predict(features)[0]
    actual = player_data["market_value"]
    is_undervalued = predicted > actual
    diff_pct = ((predicted - actual) / actual * 100) if actual != 0 else 0

    st.markdown('<div class="glass-card" style="height:100%;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;margin-bottom:18px;">'
        '<span class="section-title" style="border:none;padding:0;margin:0;">💰 Valuation Details</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    v1, v2 = st.columns(2)
    with v1:
        st.markdown(
            f'<div class="val-card val-cyan">'
            f'<div class="val-lbl">Actual Value</div>'
            f'<div class="val-amt">{fmt_money(actual)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with v2:
        st.markdown(
            f'<div class="val-card val-purple">'
            f'<div class="val-lbl">AI Predicted</div>'
            f'<div class="val-amt">{fmt_money(predicted)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    if is_undervalued:
        badge_cls = "badge-uv"
        badge_txt = f"📈 Undervalued  (+{abs(diff_pct):.1f}%)"
        rec_msg = "🟢 **Recruitment Recommended** — Model values player above current market price."
    else:
        badge_cls = "badge-ov"
        badge_txt = f"📉 Overvalued  ({diff_pct:.1f}%)"
        rec_msg = "🔵 **Fair / Over-priced** — Model values player below current market price."

    st.markdown(
        f'<div style="text-align:center;margin-top:12px;">'
        f'<span class="{badge_cls}">{badge_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if is_undervalued:
        st.success(rec_msg)
    else:
        st.info(rec_msg)

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SIMILAR PLAYERS
# ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🤝 Similar Players</div>', unsafe_allow_html=True)

sim_features = df[["goals_per90", "assists_per90"]]
sim_matrix = cosine_similarity(sim_features)
idx = df[df["player_name"] == player].index[0]
scores = list(enumerate(sim_matrix[idx]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)

rank_labels = ["1st", "2nd", "3rd", "4th", "5th"]
cols = st.columns(5)

for rank, (col, (i, score)) in enumerate(zip(cols, scores[1:6])):
    p = df.iloc[i]
    with col:
        st.markdown(
            f'<div class="sim-card">'
            f'<div class="sim-rank">#{rank_labels[rank]} Match</div>'
            f'<div class="sim-name">{p["player_name"]}</div>'
            f'<div class="sim-meta">{p["position"]}<br>{p["current_club_name"]}</div>'
            f'<div class="sim-score">{score:.0%} similarity</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br><br>", unsafe_allow_html=True)