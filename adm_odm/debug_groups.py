import pandas as pd
import sys
p = r"Y:\Study\repos\adm\anomalyDetection\adm_odm\synthetic_insights_weekly_1y_preprocessed.csv"
print('Reading', p)
df = pd.read_csv(p, parse_dates=['insight_date_time'])
print('Rows:', len(df))
print('Columns:', list(df.columns))
print(df.head().to_string(index=False))
try:
    df['week_start'] = pd.to_datetime(df['insight_date_time']).dt.to_period('W-MON').dt.start_time.dt.date
    g = df.groupby(['insight_type','target_key','week_start']).size().reset_index(name='count')
    groups = list(g.groupby(['insight_type','target_key']))
    print('Found groups:', len(groups))
    print('First 5 group keys:', [gr[0] for gr in groups[:5]])
    for i, (k, grp) in enumerate(groups[:3]):
        print('\nGroup', i+1, 'key:', k)
        print(grp.head().to_string(index=False))
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Error:', e)

