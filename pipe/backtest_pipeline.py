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
    strat_version: str
    input_filename: str
    buy_nlp_long: Optional[str] = 'go_long'
    buy_nlp_short: Optional[str] = 'go_short'

    
    def add_buy_sell(self):
        df = pd.read_csv(f'{self.input_filename}')
        print(f'data length -->: {len(df)}')
        print(df['exit_reason'].value_counts())
        cols = df.columns
        print(cols)
        for _, row in df.iterrows():
            if pd.isna(row.values).any():
                print(row.value)

        df.drop(['Unnamed: 0', 'pair'], axis=1, inplace=True)

        if conditions := [
            df['imit-enter-short (entry)'] == self.buy_nlp_long,
            df['imit-exit-short (entry)'] == self.buy_nlp_long,
            df['nlp-enter-long (entry)'] == self.buy_nlp_long,
            df['volume (entry)'] > 0,
        ]:
            df.loc[
                reduce(lambda x, y: x & y, conditions),
                ['enter_short', 'action']] = (1, 'go_short')

        if conditions2 := [
            df['imit-enter-short (entry)'] == self.buy_nlp_short,
            df['imit-exit-short (entry)'] == self.buy_nlp_short,
            df['nlp-enter-long (entry)'] == self.buy_nlp_long,
            df['volume (entry)'] > 0,
        ]:
            df.loc[
                reduce(lambda x, y: x & y, conditions2),
                ['enter_long', 'action']] = (-1, 'go_long')

        if conditions3 := [
            #df['profit_abs'] <= 0,
            #df['volume (entry)'] > 0,
            df['enter_short'] != 1.0,
            df['enter_long'] != -1.0
        ]:
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
            'long_jcrosk (entry)', 'short_kdj (entry)', \
            'imit-enter-short (entry)', 'sma-05 (entry)', 'sma-07 (entry)', \
            'imit-exit-short (entry)', 'nlp-enter-long (entry)', \
            'nlp-enter-short (entry)', 'profit_abs', 'enter_reason', 'exit_reason','action'
        ]
        print(raw_data.exit_reason.value_counts())
        return raw_data[selected_cols]
    
    def export_results_file (self):
        data = self.select_cols()
        return data.to_csv(f'./clean_data/imitate_{self.strat_version}.csv')


indicators_file = './indicators/indicators_159nlp.csv' #'/home/defi/Desktop/portfolio/projects/python/jupyter/spreadsheets/indicators.csv'

def main(strat_version: str, input_filename: Optional[str] = indicators_file):
    return ProcessData(strat_version, input_filename=indicators_file).export_results_file()


if __name__ == '__main__':
    fire.Fire(main)