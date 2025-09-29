import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the CSV file
csv_path = 'synthetic_data/synthetic_insights_weekly_1y.csv'
df = pd.read_csv(csv_path)

# Group by target_type, insight_type, and target_key, then count occurrences
count_df = df.groupby(['target_type', 'insight_type', 'target_key']).size().reset_index(name='count')

# Encode categorical features
encoded = pd.get_dummies(count_df[['target_type', 'insight_type']])
X = pd.concat([encoded, count_df['count']], axis=1)

# Fit Isolation Forest for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
count_df['anomaly'] = model.fit_predict(X)

# Show anomalies
anomalies = count_df[count_df['anomaly'] == -1]
print('Anomalies detected:')
print(anomalies)
