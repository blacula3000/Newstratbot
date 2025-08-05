# Create symbolic link for Nginx config
sudo ln -s /etc/nginx/sites-available/tradingbot /etc/nginx/sites-enabled

# Test Nginx configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx

# Start the trading bot service
sudo systemctl start tradingbot
sudo systemctl enable tradingbot

# Check status
sudo systemctl status tradingbot 