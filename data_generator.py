import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

def generate_transactions(n_samples=10000):
    transaction_id = [f"TXN{str(i).zfill(6)}" for i in range(n_samples)]
    amount = np.random.exponential(scale=150, size=n_samples).round(2)
    amount = np.clip(amount, 1, 15000)
    hour_weights = np.array([0.5,0.3,0.2,0.2,0.2,0.4,1,2,3,4,4,4,4,4,4,3,3,3,2,2,1.5,1,1,0.7], dtype=float)
    hour_weights = hour_weights / hour_weights.sum()
    hours = np.random.choice(range(24), size=n_samples, p=hour_weights)
    day_of_week = np.random.randint(0, 7, size=n_samples)
    merchant_cats = ["grocery","restaurant","electronics","travel","online","gas","retail"]
    merchant = np.random.choice(merchant_cats, size=n_samples)
    countries = ["PK","US","UK","DE","IN","AE","CN","NG"]
    country = np.random.choice(countries, size=n_samples, p=[0.35,0.2,0.1,0.05,0.1,0.05,0.1,0.05])
    prev_txn_mins = np.random.exponential(scale=300, size=n_samples)
    card_age_days = np.random.randint(1, 3650, size=n_samples)
    failed_attempts = np.random.poisson(0.2, size=n_samples)
    distance_km = np.abs(np.random.normal(50, 200, size=n_samples))
    is_online = (merchant == "online").astype(int)
    prob = np.full(n_samples, 0.03)
    prob += (amount > 1000) * 0.15
    prob += (amount > 5000) * 0.20
    prob += ((hours < 4) | (hours > 22)) * 0.12
    prob += (prev_txn_mins < 2) * 0.20
    prob += (card_age_days < 7) * 0.25
    prob += (failed_attempts >= 2) * 0.30
    prob += (distance_km > 500) * 0.15
    prob += is_online * 0.05
    prob += np.isin(country, ["NG","CN"]) * 0.08
    prob += np.isin(merchant, ["electronics","travel"]) * 0.05
    prob = np.clip(prob, 0, 0.95)
    is_fraud = (np.random.rand(n_samples) < prob).astype(int)
    df = pd.DataFrame({
        "transaction_id": transaction_id,
        "amount": amount,
        "hour": hours,
        "day_of_week": day_of_week,
        "merchant_category": merchant,
        "country": country,
        "prev_txn_mins": prev_txn_mins.round(1),
        "card_age_days": card_age_days,
        "failed_attempts": failed_attempts,
        "distance_km": distance_km.round(1),
        "is_online": is_online,
        "is_fraud": is_fraud,
    })
    return df

def encode_categoricals(df):
    df = df.copy()
    for col in ["merchant_category", "country"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

if __name__ == "__main__":
    df = generate_transactions(10000)
    print(df.head())
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Shape: {df.shape}")
