#!/bin/bash
cd /home/ubuntu/trading-bot
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart tradingbot 