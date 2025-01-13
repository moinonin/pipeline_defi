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
    buy_long_large_rl: Optional[str] = 'go_long'
    buy_long_small_rl: Optional[str] = 'go_long'
    buy_short_small_rl: Optional[str] = 'do_nothing'
    buy_short_larg_rl: Optional[str] = 'do_nothing'

    
    def add_buy_sell(self):
        df = pd.read_csv(f'{self.input_filename}')
        print(f'data length -->: {len(df)}')
        print(df['exit_reason'].value_counts())
        cols = df.columns
        #self.columns = cols
        print(cols)
        for _, row in df.iterrows():
            if pd.isna(row.values).any():
                print(row.value)

        df.drop(['Unnamed: 0', 'pair'], axis=1, inplace=True)

        if conditions := [
            #df['long-large-rl (entry)'] != self.buy_long_large_rl,
            #df['long-small-rl (entry)'] != self.buy_long_small_rl,
            df['first-long (entry)'] == True,
            df['second-long (entry)'] == True,
            df['short-sig (entry)'] != 1,
            df['profit_abs'] > 0,
            df['volume (entry)'] > 0,
        ]:
            df.loc[
                reduce(lambda x, y: x & y, conditions),
                ['enter_long', 'action']] = (1, 'go_long')

        if conditions2 := [
            #df['short-small-rl (entry)'] == self.buy_short_small_rl,
            #df['short-large-rl (entry)'] == self.buy_short_larg_rl,
            df['first-short (entry)'] == True,
            df['second-short (entry)'] == True,
            df['long-sig (entry)'] != 1,
            df['profit_abs'] > 0,
            df['volume (entry)'] > 0,
        ]:
            df.loc[
                reduce(lambda x, y: x & y, conditions2),
                ['enter_short', 'action']] = (-1, 'go_short')

        if conditions3 := [
            df['profit_abs'] <= 0,
            #df['enter_short'] != -1.0,
            #df['enter_long'] != 1.0
            df['volume (entry)'] > 0
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
            'nlp-enter-short (entry)', 'profit_abs', 'enter_reason', \
            'long-small-rl (entry)', 'short-small-rl (entry)', 'long-large-rl (entry)', \
            'short-large-rl (entry)', 'exit_reason','action'
        ]
        print(raw_data.exit_reason.value_counts())
        found_cols = []
        for idx, col in enumerate(selected_cols):
            if col not in raw_data.columns:
                print(f"Column {idx}: {col} not found! ignoring!")
            else:
                found_cols.append(col)

        return raw_data[found_cols]
    
    def export_results_file (self):
        data = self.select_cols()
        return data.to_csv(f'./clean_data/imitate_{self.strat_version}.csv')


#indicators_file = './indicators/indicators_159nlp.csv' #'/home/defi/Desktop/portfolio/projects/python/jupyter/spreadsheets/indicators.csv'

def main(strat_version: str, input_filename: Optional[str] = None):
    indicators_file = f'./indicators/indicators_{strat_version}.csv'
    return ProcessData(strat_version, input_filename=indicators_file).export_results_file()


if __name__ == '__main__':
    fire.Fire(main)