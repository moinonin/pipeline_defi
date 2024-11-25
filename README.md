### Usage
The this application ingests backtest results to build machine learning models using scikit learn.  
In order to use it, the results must come from freqtrade backtesting with the following exported signals extra arguments:  
```
docker-compose run --rm freqtrade backtesting-analysis -c user_data/configrotichUSDT10.json --analysis-to-csv --analysis-groups 0 2 \n
--enter-reason-list first_buy second_buy --exit-reason-list roi short-ml-take-profit long-ml-take-profit short-risk-profit force-exit \n
stop_loss long-ml-exit short-ml-exit force_exit liquidation --indicator-list open high ema-26 ema-12 low mean-grad-hist close volume \n
sma-25 long_jcrosk short_kdj profit_ratio profit_abs nlp-enter-long nlp-enter-short nlp-exit-long nlp-exit-short long-entery-gradient \n
short-entry-gradient grads-ratio imit-enter-long imit-enter-short imit-exit-long imit-exit-short
```
