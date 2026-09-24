import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000  

def generate_dataset(n=N):
    pages_viewed = np.random.poisson(lam=5, size=n) + 1
    time_spent_sec = np.random.gamma(shape=2.0, scale=60, size=n) 
    previous_visits = np.random.poisson(lam=2, size=n)
    products_viewed = np.random.poisson(lam=3, size=n)
    added_to_cart = np.random.binomial(1, 0.3, size=n)
    device_type = np.random.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])
    previous_purchases = np.random.poisson(lam=1, size=n)

    z = (
        0.18 * pages_viewed
        + 0.02 * time_spent_sec
        + 0.35 * previous_visits
        + 0.10 * products_viewed
        + 2.20 * added_to_cart
        + 0.90 * (previous_purchases > 0)
        + 0.30 * previous_purchases
        - 7.3  
    )

    device_effect = np.where(device_type == "desktop", 0.3,
                     np.where(device_type == "tablet", 0.0, -0.15))
    z = z + device_effect

    prob = 1 / (1 + np.exp(-z))
    purchase = np.random.binomial(1, prob)

    df = pd.DataFrame({
        "pages_viewed": pages_viewed,
        "time_spent_sec": np.round(time_spent_sec, 1),
        "previous_visits": previous_visits,
        "products_viewed": products_viewed,
        "added_to_cart": added_to_cart,
        "device_type": device_type,
        "previous_purchases": previous_purchases,
        "purchase": purchase
    })
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/customer_data.csv", index=False)
    print(f"Generated {len(df)} rows -> data/customer_data.csv")
    print(df["purchase"].value_counts(normalize=True))
