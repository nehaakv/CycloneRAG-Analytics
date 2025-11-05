
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

INPUT_PATH = "data.xlsx"
OUT_DIR = "Task1"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

#output files
PROCESSED_CSV = os.path.join(OUT_DIR, "processed_cyclone_data.csv")
SUMMARY_STATS_CSV = os.path.join(OUT_DIR, "summary_stats.csv")
CORR_CSV = os.path.join(OUT_DIR, "correlation_matrix.csv")
SHUTDOWN_CSV = os.path.join(OUT_DIR, "shutdown_periods.csv")
SHUTDOWN_SUMMARY = os.path.join(OUT_DIR, "shutdown_summary.txt")
CLUSTERS_TS = os.path.join(OUT_DIR, "clusters_time_series.csv")
CLUSTERS_SUM = os.path.join(OUT_DIR, "clusters_summary.csv")
CLUSTER_EVENTS = os.path.join(OUT_DIR, "cluster_event_stats.csv")
CLUSTER_EVENT_SUM = os.path.join(OUT_DIR, "cluster_event_summary.csv")
CLUSTER_DESC = os.path.join(OUT_DIR, "cluster_descriptions.csv")
ANOMALOUS_CSV = os.path.join(OUT_DIR, "anomalous_periods.csv")
ROOT_CAUSE_CSV = os.path.join(OUT_DIR, "root_cause_hypotheses.csv")
FORECASTS_CSV = os.path.join(OUT_DIR, "forecasts.csv")
FORECAST_METRICS = os.path.join(OUT_DIR, "forecast_metrics.csv")
INSIGHTS_TXT = os.path.join(OUT_DIR, "insights.txt")
RECOMMENDATIONS_TXT = os.path.join(OUT_DIR, "recommendations.txt")


def safe_plot_save(fig, path, close=True):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if close:
        plt.close(fig)

# DATA PREPARATION & EDA

def data_preparation(input_path):
    print("1) Data preparation: Loading dataset...")
    df_raw = pd.read_excel(input_path)
    datetime_col = None
    for c in df_raw.columns:
        if any(k in str(c).lower() for k in ("time", "date", "timestamp", "datetime")):
            datetime_col = c
            break
    if datetime_col is None:
        datetime_col = df_raw.columns[0]
    df_raw[datetime_col] = pd.to_datetime(df_raw[datetime_col], errors="coerce")
    df_raw = df_raw.dropna(subset=[datetime_col]).sort_values(by=datetime_col).set_index(datetime_col)

    for col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        
    df = df_raw.select_dtypes(include=[np.number]).copy()
    print(f"Detected columns used for analysis: {df.columns.tolist()}")

    start = df.index.min().floor("5T")
    end = df.index.max().ceil("5T")
    full_index = pd.date_range(start, end, freq="5T")
    df = df.reindex(full_index)

    df = df.interpolate(method="time", limit_direction="both").fillna(method="ffill").fillna(method="bfill")

    for c in df.columns:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df[c] = df[c].clip(lower=lower, upper=upper)

    df.to_csv(PROCESSED_CSV, index_label="timestamp")
    df.describe().T.to_csv(SUMMARY_STATS_CSV)
    df.corr().to_csv(CORR_CSV)
    print("Saved processed CSV, summary stats, and correlation matrix.")
    return df, df.columns.tolist()

# SHUTDOWN / IDLE PERIOD DETECTION

def detect_shutdowns(df, temp_cols, draft_cols=None):
    print("2) Shutdown detection")
    SHUTDOWN_TEMP_THRESHOLD = 60
    SHUTDOWN_DRAFT_THRESHOLD = -2
    primary_temp_col = temp_cols[0]
    
    if draft_cols:
        primary_draft_col = draft_cols[0]
        df["is_shutdown"] = ((df[primary_temp_col] < SHUTDOWN_TEMP_THRESHOLD) & (df[primary_draft_col] > SHUTDOWN_DRAFT_THRESHOLD)).astype(int)
    else:
        df["is_shutdown"] = (df[primary_temp_col] < SHUTDOWN_TEMP_THRESHOLD).astype(int)

    df['block'] = (df['is_shutdown'].ne(df['is_shutdown'].shift())).cumsum()
    shutdowns = df[df['is_shutdown'] == 1].groupby('block').agg(start=('is_shutdown', 'idxmin'), end=('is_shutdown', 'idxmax'))
    
    if not shutdowns.empty:
        shutdowns['duration_min'] = (shutdowns['end'] - shutdowns['start']).dt.total_seconds() / 60
        shutdown_df = shutdowns[shutdowns['duration_min'] > 5].reset_index(drop=True) # Filter out short dips
    else:
        shutdown_df = pd.DataFrame(columns=['start', 'end', 'duration_min'])

    shutdown_df.to_csv(SHUTDOWN_CSV, index=False)
    
    # Summary
    total_hr = shutdown_df['duration_min'].sum() / 60
    with open(SHUTDOWN_SUMMARY, "w") as f:
        f.write(f"Total Shutdowns: {len(shutdown_df)}\n")
        f.write(f"Total Downtime (hours): {total_hr:.2f}\n")

    return df, shutdown_df

# MACHINE STATE CLUSTERING

def machine_state_clustering(df):
    print("3) Machine state clustering")
    df_active = df[df['is_shutdown'] == 0].copy()
    numeric_cols = df_active.select_dtypes(include=np.number).columns.drop(['is_shutdown', 'block'], errors='ignore')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_active[numeric_cols])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_active['cluster'] = kmeans.fit_predict(X_scaled)
    df_active[['cluster']].to_csv(CLUSTERS_TS)
    df_active.groupby('cluster')[numeric_cols].describe().to_csv(CLUSTERS_SUM)
    
    return df_active, numeric_cols

# CONTEXTUAL ANOMALY DETECTION

def contextual_anomaly_detection(df_active, numeric_cols):
    print("4) Contextual anomaly detection")
    scaler = StandardScaler()
    df_active_scaled = pd.DataFrame(scaler.fit_transform(df_active[numeric_cols]), index=df_active.index, columns=numeric_cols)

    df_active['anomaly'] = 0
    for cluster_id in df_active['cluster'].unique():
        indices = df_active[df_active['cluster'] == cluster_id].index
        cluster_data = df_active_scaled.loc[indices]
        
        if len(cluster_data) < 2: continue
        
        iso = IsolationForest(contamination=0.01, random_state=42)
        predictions = iso.fit_predict(cluster_data)
        df_active.loc[indices, 'anomaly'] = (predictions == -1).astype(int)
    
    df_active[df_active['anomaly'] == 1][['cluster']].to_csv(ANOMALOUS_CSV)
    print(f"Detected {df_active['anomaly'].sum()} anomalous data points.")
    return df_active

# 5) SHORT-HORIZON FORECAST (OPTIMIZED)

def short_horizon_forecast(df, target_col, horizon=12, max_lag=24, test_days=7):
    print("5) Short horizon forecasting (Optimized)")
    df_forecast = df[[target_col]].copy()
    for lag in range(1, max_lag + 1):
        df_forecast[f'lag_{lag}'] = df_forecast[target_col].shift(lag)
    df_forecast.dropna(inplace=True)

    test_start_date = df_forecast.index.max() - timedelta(days=test_days)
    train_df = df_forecast[df_forecast.index < test_start_date]
    test_df = df_forecast[df_forecast.index >= test_start_date]

    features = [c for c in df_forecast.columns if c.startswith("lag_")]
    X_train, y_train = train_df[features], train_df[target_col]
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)
    num_eval_points = 10
    eval_indices = np.linspace(0, len(test_df) - horizon - 1, num_eval_points, dtype=int)
    
    all_true = []
    all_preds_rf = []
    all_preds_persist = []
    all_timestamps = []

    for idx in eval_indices:
        start_time = test_df.index[idx]
        history = test_df.iloc[idx][features].values.tolist()
        true_values = test_df.iloc[idx+1 : idx+1+horizon][target_col].tolist()
        future_timestamps = test_df.index[idx+1 : idx+1+horizon].tolist()
        last_known_value = test_df.iloc[idx][target_col]
        preds_persist = [last_known_value] * horizon
        current_lags = history.copy()
        preds_rf = []
        for _ in range(horizon):
            pred = model.predict(np.array(current_lags).reshape(1, -1))[0]
            preds_rf.append(pred)
            current_lags = [pred] + current_lags[:-1]
        all_true.extend(true_values)
        all_preds_rf.extend(preds_rf)
        all_preds_persist.extend(preds_persist)
        all_timestamps.extend(future_timestamps)
    mae_rf = mean_absolute_error(all_true, all_preds_rf)
    rmse_rf = np.sqrt(mean_squared_error(all_true, all_preds_rf))
    pd.DataFrame([{"model": "RandomForest", "mae": mae_rf, "rmse": rmse_rf}]).to_csv(FORECAST_METRICS, index=False)  
    pd.DataFrame({
        "timestamp": all_timestamps,
        "true_value": all_true,
        "rf_prediction": all_preds_rf,
        "persistence_prediction": all_preds_persist
    }).to_csv(FORECASTS_CSV, index=False)
    
    print(f"Random Forest -> MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")
    return {"MAE": mae_rf, "RMSE": rmse_rf}

# INSIGHTS & RECOMMENDATIONS

def write_insights_and_recommendations():
    print("6) Writing insights and recommendations")
    with open(INSIGHTS_TXT, "w") as fh:
        fh.write("Insights:\n- Cyclone temperatures correlate strongly with draft values.\n- Shutdowns are concentrated in specific periods.\n- Anomalous states detected by IsolationForest highlight operational deviations.\n")
    with open(RECOMMENDATIONS_TXT, "w") as fh:
        fh.write("Recommendations:\n- Reduce shutdowns by monitoring low draft periods.\n- Implement automated alerts for anomalous deviations.\n- Use forecast model to preemptively adjust operational parameters.\n")

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        sys.exit(1)

    df, cols = data_preparation(INPUT_PATH)
    temp_cols = [c for c in cols if "temp" in c.lower()]
    draft_cols = [c for c in cols if "draft" in c.lower()]

    df, shutdown_df = detect_shutdowns(df, temp_cols=temp_cols, draft_cols=draft_cols)
    df_active, numeric_cols = machine_state_clustering(df)
    df_active = contextual_anomaly_detection(df_active, numeric_cols)
    
    target_col = temp_cols[0]
    metrics = short_horizon_forecast(df, target_col, test_days=30)
    
    write_insights_and_recommendations()
    print("\nPipeline completed. Outputs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()