import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_revenue_model(df):
    features = [
        "impressions",
        "clicks",
        "conversions",
        "ad_spend",
        "retention_rate",
        "customer_ltv",
        "ctr",
        "conversion_rate",
        "cac",
        "roas"
    ]

    target = "revenue"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        by="importance",
        ascending=False
    )

    return {
        "model": model,
        "mae": mae,
        "r2": r2,
        "feature_importance": feature_importance
    }