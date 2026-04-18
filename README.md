# ⚽ Player Recruitment AI

A data-driven machine learning application to optimize football (soccer) player recruitment. This application leverages player performance data to:
- Identify player roles using clustering.
- Predict player market values to evaluate if a player is under/overvalued.
- Find similar players using cosine similarity.
- Provide an interactive Streamlit dashboard for scouting and analysis.

## Features
- **Data Engineering**: Cleans and merges player profiles, match appearances, and market valuations.
- **Role Clustering**: Groups players into clusters based on their play styles and stats.
- **Valuation Prediction**: A machine learning model to estimate fair market values and highlight undervalued gems.
- **Similarity Search**: Recommends players with similar statistical profiles.
- **Dashboard**: A clean, modern Streamlit UI to filter and analyze the scouting pool.

## Project Structure
- `main.py` - Pipeline entry point to process data, train models, and generate the final dataset.
- `app/streamlit_app.py` - Streamlit dashboard application.
- `src/` - Core logic for preprocessing, clustering, similarity, and prediction models.
- `data/` and `models/` - Directories for generated datasets and trained machine learning artifacts (generated after running `main.py`).

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the data pipeline to generate the dataset and models:
   ```bash
   python main.py
   ```
3. Start the dashboard:
   ```bash
   streamlit run app/streamlit_app.py
   ```
