# for file system import
#from imit_main import imit_signal as imit # for local file import
#from nlp_main import nlp_signal as nlp # for local file import

# for install import
from nlp.imit_main import imit_signal as imit # install import
from nlp.nlp_main import nlp_signal as nlp # install import

import numpy as np
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix

def getSignal(*args):
    try:
        return imit(*args)
    except Exception as e:
        print(e)


data = pd.read_csv('/home/defi/Desktop/portfolio/projects/python/pipeline_defi/clean_data/imitate_159nlp.csv') #  ../jupyter/spreadsheets/tuned-indicators.csv
#print(data.head())

def prep_data():
    train_data = pd.DataFrame()
    for col in data.columns:
        col_name = col.split(' ')[0]
        train_data[f'{col_name}'] = data[col]

    return train_data

#df = prep_data()

df = data
#print(df['exit_reason'].value_counts())

#df.drop(['open_date'], inplace=True, axis=1)
#df['short_kdj'] = df['short_kdj'].astype(int)
print(df.head())

# Filter what you want to test
#df = df[df['profit_abs'] < 0]
#df = df[df['exit_reason'] == 'long-exit-profit']
#df = df[df['enter_reason'] == 'first_buy']
#df = df.tail(10)
print(df.columns)


df['imit-action'] = df.apply(lambda row: getSignal( row['sma-05 (entry)'],row['sma-07 (entry)'],row['sma-25 (entry)']), axis=1
                    )

# Convert Series to list
#print(df.head())

#string_series = pd.Series(df['buy-imit-long_go_long'].tolist())
string_series = pd.Series(df['action'].tolist())

# Function to create sliding windows
def sliding_windows(series=string_series, window_size=3):
    for i in range(len(series) - window_size + 1):
        yield series[i:i + window_size]

predictions = [np.nan] * len(string_series)  # Start with NaN, predictions will overwrite later

# Create sliding windows and predict
window_size = 3  # Sliding window size of 5
for i, window in enumerate(sliding_windows(series=string_series, window_size=3)):
    #print(window)
    pred = nlp(window)  # Replace with your actual model's predict function
    predictions[i + window_size - 1] = pred

df['nlpreds'] = predictions
df['nlpreds'] = df['nlpreds'].bfill()#fillna(method='bfill')
accuracy = 0 if len(df) == 0 else len(df[df['action'] == df['nlpreds']]) * 100/len(df)
print(f'accuracy: --> {accuracy} %')
df['reward'] = df['profit_abs']
df.loc[:,'is_short'] = np.where(df['enter_reason'] == 'first_buy', 0, 1)

if 'sma-compare' not in df.columns:
    df['sma-compare (entry)'] = ((df['sma-07 (entry)'] > df['sma-05 (entry)']) & (df['sma-25 (entry)'] > df['sma-07 (entry)'])).astype(int)

print(df.head())
print(' ')
print(df['sma-compare (entry)'].value_counts())
print(' ')
print(df['nlpreds'].value_counts())
print(' ')
print(df['imit-action'].value_counts())
print(' ')
print(df['action'].value_counts())
# Get confusion matrix
cm = confusion_matrix(df['action'], df['nlpreds'])

print("Confusion Matrix")
print(cm)

col_selection = ['sma-05 (entry)', 'sma-07 (entry)', 'sma-25 (entry)', \
                'sma-compare (entry)','is_short', 'action', 'imit-action', 'nlpreds','reward'
            ]

train_data = df[col_selection]

print(train_data.head())

train_data.to_csv('../jupyter/spreadsheets/rlhf_159nlp2.csv')
