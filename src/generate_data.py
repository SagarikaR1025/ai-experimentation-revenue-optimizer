import pandas as pd
import numpy as np
import os

np.random.seed(42)

os.makedirs("data", exist_ok=True)

n = 1000

campaigns = ["Campaign_A", "Campaign_B", "Campaign_C"]

data = pd.DataFrame({
    "campaign": np.random.choice(campaigns, n),

    "impressions": np.random.randint(1000, 10000, n),

    "clicks": np.random.randint(100, 2000, n),

    "conversions": np.random.randint(10, 300, n),

    "ad_spend": np.random.uniform(500, 10000, n),

    "revenue": np.random.uniform(1000, 50000, n),

    "retention_rate": np.random.uniform(0.4, 0.95, n),

    "customer_ltv": np.random.uniform(50, 5000, n)
})

# Derived metrics
data["ctr"] = data["clicks"] / data["impressions"]

data["conversion_rate"] = data["conversions"] / data["clicks"]

data["cac"] = data["ad_spend"] / data["conversions"]

data["roas"] = data["revenue"] / data["ad_spend"]

# Save dataset
data.to_csv("data/marketing_experiments.csv", index=False)

print("Dataset generated successfully!")