# Restart the bot
sudo supervisorctl restart tradingbot

# Check bot status
sudo supervisorctl status tradingbot

# View real-time logs
sudo supervisorctl tail -f tradingbot

# Update bot code
cd ~/trading-bot
git pull  # if using git
source venv/bin/activate
pip install -r requirements.txt
sudo supervisorctl restart tradingbot 