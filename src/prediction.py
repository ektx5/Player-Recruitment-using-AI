from sklearn.ensemble import RandomForestRegressor
import joblib

def train_price_model(df):
    X = df[['age', 'goals_per90', 'assists_per90', 'role_cluster']]
    y = df['market_value']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'models/price_model.pkl')

    return model