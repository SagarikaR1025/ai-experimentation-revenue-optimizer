import pandas as pd
from scipy.stats import ttest_ind


def compare_campaigns(df, metric="conversion_rate"):

    campaigns = df["campaign"].unique()

    if len(campaigns) < 2:
        return None

    campaign_1 = df[df["campaign"] == campaigns[0]][metric]

    campaign_2 = df[df["campaign"] == campaigns[1]][metric]

    t_stat, p_value = ttest_ind(campaign_1, campaign_2)

    return {
        "campaign_1": campaigns[0],
        "campaign_2": campaigns[1],
        "metric": metric,
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05
    }