import pandas as pd
import numpy as np
from dataclasses import dataclass
from pandas import DataFrame, Series
from functools import reduce
import fire
from typing import Optional

# Receive and process the csv file with backtest signal candles
@dataclass
class ProcessData:
    version: str
    input_filename: str
    buy_nlp_long: Optional[str] = 'go_long'
    buy_nlp_short: Optional[str] = 'go_short'
    
    def add_buy_sell(self):
        df = pd.read_csv(f'{self.input_filename}')
        print(df['profit_abs'].head())
        cols = df.columns
        print(cols)
        for index, row in df.iterrows():
            if pd.isna(row.values).any():
                print(row.value)

        '''
        reject_cols = ['stop_loss']

        for col in cols:
            if col in reject_cols:
                df = df[df['exit_reason'] != f'{col}']
        '''
        df.drop(['Unnamed: 0', 'pair'], axis=1, inplace=True)

        conditions = []

        conditions.append(
                df['nlp-enter-long (entry)'] == self.buy_nlp_long
            )
        conditions.append(df['nlp-enter-short (entry)'] != 'do_nothing')
        conditions.append(df['profit_abs'] > 0)
        conditions.append(df['volume (entry)'] > 0)
        
        if conditions:
            df.loc[
                reduce(lambda x, y: x & y, conditions),
                ['enter_long', 'action']] = (1, 'go_long')

        conditions2 = []

        conditions2.append(
                df['nlp-enter-short (entry)'] == self.buy_nlp_short
            )
        conditions2.append(df['nlp-enter-long (entry)'] != 'do_nothing')
        conditions2.append(df['profit_abs'] > 0)
        conditions2.append(df['volume (entry)'] > 0)
        
        if conditions2:
            df.loc[
                reduce(lambda x, y: x & y, conditions2),
                ['enter_short', 'action']] = (-1, 'go_short')

        conditions3 = []

        '''
        conditions3.append(
            (df['grads-ratio (entry)'] <=   1.0) &
            (df['grads-ratio (entry)'] >=   -3.0)
        )
        '''
        conditions3.append(df['profit_ratio'] <= 0)
        #conditions3.append(df['nlp-enter-long (entry)'] == 'do_nothing')
        #conditions3.append(df['nlp-enter-short (entry)'] == 'do_nothing')
        #conditions3.append(df['exit_reason'] == 'stop_loss')
        conditions3.append(df['volume (entry)'] > 0)

        if conditions3:
            df.loc[
                reduce(lambda x, y: x & y, conditions3),
                ['enter_none', 'action']] = (0, 'do_nothing')

        df_copy = df.copy()
        short_kdj = df['short_kdj (entry)'].astype(int)
        df_copy['short_kdj (entry)'] = short_kdj
        
        return df_copy
    
    def select_cols(self):

        raw_data = self.add_buy_sell()
        
        selected_cols = [
            'open (entry)', 'high (entry)', 'ema-26 (entry)', \
            'ema-12 (entry)', 'low (entry)', 'mean-grad-hist (entry)', \
            'close (entry)', 'volume (entry)', 'sma-25 (entry)', \
            'long_jcrosk (entry)', 'short_kdj (entry)', 'action'
        ]

        cleaned_data = raw_data[selected_cols]
        
        return cleaned_data
    
    def export_results_file (self):
        data = self.select_cols()
        return data.to_csv(f'./clean_data/imitate_{self.version}.csv')


indicators_file = './indicators/indicators.csv' #'/home/defi/Desktop/portfolio/projects/python/jupyter/spreadsheets/indicators.csv'

def main(version: str, input_filename: Optional[str] = indicators_file):
    return ProcessData(version, input_filename=indicators_file).export_results_file()


if __name__ == '__main__':
    fire.Fire(main)