# Step 1: Launch EC2 Instance
# ------------------------
# - Go to AWS Console
# - Launch new EC2 instance
# - Choose Ubuntu Server 20.04 LTS
# - Select t2.micro (free tier) or t2.small
# - Configure Security Group:
#   * Allow SSH (Port 22)
#   * Allow HTTP (Port 80)
#   * Allow HTTPS (Port 443)
#   * Allow Custom TCP (Port 5000)
# - Create and download your key pair (example: trading-bot.pem)

# Step 2: Connect to Your Instance
# ------------------------
chmod 400 trading-bot.pem
ssh -i trading-bot.pem ubuntu@your-ec2-public-ip

# Step 3: Update System Packages
# ------------------------
sudo apt-get update
sudo apt-get upgrade -y

# Step 4: Install Required System Packages
# ------------------------
sudo apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools nginx git supervisor

# Step 5: Install TA-Lib
# ------------------------
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Step 6: Set Up Project Directory
# ------------------------
mkdir -p ~/trading-bot/logs
cd ~/trading-bot
python3 -m venv venv
source venv/bin/activate

# Step 7: Copy Your Code
# ------------------------
# Either clone from your repository:
# git clone https://github.com/yourusername/trading-bot.git .
# Or manually create and copy files using nano/vim

# Step 8: Install Python Dependencies
# ------------------------
pip install -r requirements.txt

# Step 9: Create Supervisor Configuration
# ------------------------
sudo nano /etc/supervisor/conf.d/tradingbot.conf

# Step 10: Configure Nginx
# ------------------------
sudo ln -s /etc/nginx/sites-available/tradingbot /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx

# Step 11: Start Services
# ------------------------
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start tradingbot

# Step 12: Monitor Logs
# ------------------------
# View application logs
tail -f ~/trading-bot/logs/supervisor.out.log
tail -f ~/trading-bot/logs/supervisor.err.log

# View Nginx logs
tail -f ~/trading-bot/logs/nginx-access.log
tail -f ~/trading-bot/logs/nginx-error.log 