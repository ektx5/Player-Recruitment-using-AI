from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def train_clustering(df):
    features = df[['goals_per90', 'assists_per90']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=5, random_state=42)
    df['role_cluster'] = model.fit_predict(X_scaled)

    joblib.dump(model, 'models/clustering.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    return df