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
    buy_long_bid_rl: Optional[str] = 'go_short'
    buy_short_large_rl: Optional[str] = 'do_nothing'
    buy_long_small_rl: Optional[str] = 'go_long'
    buy_long_large_rl: Optional[str] = 'go_long'
    buy_short_bid_rl: Optional[str] = 'go_long'
    buy_short_small_rl: Optional[str] = 'go_short'
    
    sell_short_nlp_custom_exit: Optional[str] = 'go_short'
    buy_nlp_short: Optional[str] = 'go_short'

    buy_nlp_long: Optional[str] = 'go_long'
    buy_long_min_grad: Optional[float] = 0.0
    
    buy_regbot_long: Optional[int] = -1
    buy_regbot_long_count: Optional[int] = 10
    buy_long_study_candles: Optional[int] = 15

    buy_regbot_short: Optional[int] = 1
    buy_regbot_short_count: Optional[int] = 13
    buy_short_study_candles: Optional[int] = 15



    
    def add_buy_sell(self, exclude: Optional[list] = None):
        df = pd.read_csv(f'{self.input_filename}')
        # Remove liquidation and stop loss trades
        #df = df[(df['exit_reason'] != 'liquidation') & (df['exit_reason'] != 'stop_loss')]
        print(f'data length -->: {len(df)}')
        print(df['exit_reason'].value_counts())
        cols = df.columns
        #self.columns = cols
        #print(cols)
        for _, row in df.iterrows():
            if pd.isna(row.values).any():
                print(row)

        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        #df['long_signal'] = (df[f'regbot_long_{self.buy_regbot_long} (entry)'].rolling(window=self.buy_long_study_candles).apply(lambda x: x.tolist().count(self.buy_regbot_long) >= self.buy_regbot_long_count, raw=False))
        #df['short_signal'] = (df[f'regbot_short_{self.buy_regbot_short} (entry)'].rolling(window=self.buy_short_study_candles).apply(lambda x: x.tolist().count(self.buy_regbot_short) >= self.buy_regbot_short_count, raw=False))
        #df = df.dropna(subset=['short_signal', 'long_signal'])
        print(df.head())
        if conditions := [
            #df['long-large-rl (entry)'] == self.buy_long_large_rl,
            df['long-bid-rl (entry)'] == self.buy_long_bid_rl,
            #df['long-small-rl (entry)'] == self.buy_long_small_rl,
            #df['nlp-exit-short (entry)'] == self.sell_short_nlp_custom_exit,
            #df['grads-ratio (entry)'] < self.buy_long_min_grad,
            #df['imit-enter-short (entry)'] == self.buy_nlp_long,
            #df['imit-exit-short (entry)'] == self.buy_nlp_long,
            #df['nlp-enter-long (entry)'] == self.buy_nlp_long,
            #df['first-long (entry)'] == True,
            #df['second-long (entry)'] == True,
            #df['short-sig (entry)'] != 1,
            #df['long_signal'] == 1,
            #df['regbot_long_-1 (entry)'] == self.buy_regbot_long,
            df['profit_abs'] > 0,
            df['volume (entry)'] > 0,
        ]:
            df.loc[
                reduce(lambda x, y: x & y, conditions),
                ['enter_long', 'action']] = (1, 'go_long')

        if conditions2 := [
            #df['short-large-rl (entry)'] == self.buy_short_large_rl,
            #df['short-small-rl (entry)'] == self.buy_short_small_rl,
            #df['short-large-rl (entry)'] == self.buy_short_large_rl,
            df['short-bid-rl (entry)'] == self.buy_short_bid_rl,
            #df['long-small-rl (entry)'] != self.buy_long_small_rl,
            #df['imit-enter-short (entry)'] == self.buy_nlp_short,
            #df['imit-exit-short (entry)'] == self.buy_nlp_short,
            #df['nlp-enter-short (entry)'] == self.buy_nlp_short,
            #df['first-short (entry)'] == True,
            #df['second-short (entry)'] == True,
            #df['long-sig (entry)'] != 1,
            #df['short_signal'] == 1,
            #df['regbot_short_1 (entry)'] == self.buy_regbot_short,
            df['profit_abs'] > 0,
            df['volume (entry)'] > 0
        ]:
            df.loc[
                reduce(lambda x, y: x & y, conditions2),
                ['enter_short', 'action']] = (-1, 'go_short')

        if conditions3 := [
            df['profit_abs'] <= 0,
            df['enter_short'] != -1.0,
            df['enter_long'] != 1.0,
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
        #print(raw_data.columns)
        selected_cols = [
            'pair','open_date','open (entry)', 'high (entry)', 'ema-26 (entry)', \
            'ema-12 (entry)', 'low (entry)', 'mean-grad-hist (entry)', \
            'close (entry)', 'volume (entry)', 'sma-25 (entry)', \
            'long_jcrosk (entry)', 'short_kdj (entry)', \
            'imit-enter-short (entry)', 'sma-05 (entry)', 'sma-07 (entry)', \
            'imit-exit-short (entry)', 'nlp-enter-long (entry)', \
            'nlp-enter-short (entry)', 'profit_abs', 'enter_reason', \
            'long-small-rl (entry)', 'short-small-rl (entry)', 'long-large-rl (entry)', \
            'short-large-rl (entry)', 'long-bid-rl (entry)', 'short-bid-rl (entry)', 'exit_reason','action', \
            'nlp-exit-short (entry)', 'grads-ratio (entry)', 'regbot_long_-1 (entry)', 'regbot_short_1 (entry)', \
            'long_signal', 'short_signal'
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
        print(data.columns)
        return data.to_csv(f'./clean_data/imitate_{self.strat_version}.csv')


#indicators_file = './indicators/indicators_159nlp.csv' #'/home/defi/Desktop/portfolio/projects/python/jupyter/spreadsheets/indicators.csv'

def main(strat_version: str, input_filename: Optional[str] = None):
    indicators_file = f'./indicators/indicators_{strat_version}.csv'
    return ProcessData(strat_version, input_filename=indicators_file).export_results_file()


if __name__ == '__main__':
    fire.Fire(main)