import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# expected_price[crop] = 0.7 * season_avg + 0.3 * (season_avg + 30 * trend)

def compute_season_avg_and_trend(df, crop_name):
    # converting the dates to datetime format
    df['arrival_date'] = pd.to_datetime(df['arrival_date'])
    today = df['arrival_date'].max()

    #filtering for the required crop
    crop_df = df[df['commodity'].str.lower().str.contains(crop_name.lower(), na=False)].copy()
    if crop_df.empty:
        return {'seasonal_avg': None, 'trend': None, 'expected_price': None}
    
    #considering past 12 months
    start_12m = today - timedelta(days=365)
    df_12m = crop_df[crop_df['arrival_date'] >= start_12m]
    seasonal_avg = df_12m['modal_price'].astype(float).mean() if not df_12m.empty else None

    #past 30 days (trend)
    start_30d = today - timedelta(days=30)
    df_30d = crop_df[crop_df['arrival_date'] >= start_30d].sort_values(by='arrival_date')
    trend= None
    if(len(df_30d) >= 2):
        X = np.arange(len(df_30d)).reshape(-1, 1)
        Y = df_30d['modal_price'].astype(float).values.reshape(-1, 1)
        model = LinearRegression().fit(X, Y)
        trend = float(model.coef_[0][0]) # Rs per day
    # expected price
    exp_price = None
    if trend is not None and seasonal_avg is not None:
        exp_price = 0.7 * seasonal_avg + 0.3 * (seasonal_avg + 30 * trend)
    return {'seasonal_avg': round(seasonal_avg, 3), 'trend': round(trend, 3), 'expected_price': round(exp_price, 3)}


'''if __name__ == "__main__":
    df = pd.read_csv("data/mandi_prices.csv")
    result = compute_season_avg_and_trend(df, "Potato")
    print("Potato: ", result)'''

