# Deep RL Portfolio Optimization (gitingest)


Reference image (project brief): `/mnt/data/f71a2043-ddc7-429b-89ba-c34ca0c1a984.png`


## Overview
Starter repository implementing a Gym-compatible trading environment and a DQN agent that considers transaction costs. The submission is organized for easy ingestion into CI/CD or grading pipelines.


## Quick start
```bash
python -m pip install -r requirements.txt
python data/download_data.py --tickers AAPL MSFT GOOGL AMZN FB TSLA NVDA JPM BAC XOM --start 2015-01-01 --end 2024-01-01
python train.py --config config.yaml
