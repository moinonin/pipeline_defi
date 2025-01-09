### Usage
The this application ingests backtest results to build machine learning models using scikit learn.  
In order to use it, the results must come from freqtrade backtesting with the following exported signals extra arguments:  
```
docker-compose run --rm freqtrade backtesting-analysis -c user_data/configrotichUSDT10.json \n
--analysis-to-csv --analysis-groups 0 2 --enter-reason-list first_buy second_buy --exit-reason-list \n
roi stop_loss force_exit long-exit-profit short-exit-profit trailing_stop_loss long-signal-on \n
short-signal-on long-ml-take-profit long-ml-exit --indicator-list open high ema-26 ema-12 \n
low mean-grad-hist close volume long_jcrosk short_kdj sma-05 sma-07 sma-25 profit_abs \n
profit_ratio sma-compare imit-enter-short imit-exit-short long-small-rl short-small-rl long-large-rl short-large-rl nlp-enter-long nlp-enter-short
```
