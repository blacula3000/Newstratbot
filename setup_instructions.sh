# 1. Launch an EC2 instance
# - Use Ubuntu Server 20.04 LTS
# - t2.micro is fine for testing
# - Configure security group to allow:
#   - SSH (port 22)
#   - HTTP (port 80)
#   - HTTPS (port 443)
#   - Custom TCP (port 5000)

# 2. SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# 3. Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# 4. Install Python and required system packages
sudo apt-get install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools nginx git

# 5. Install TA-Lib dependencies
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xvf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# 6. Clone your repository (assuming it's on GitHub)
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# 7. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 8. Install dependencies
pip install -r requirements.txt

# 9. Create a systemd service file for the bot
sudo nano /etc/systemd/system/tradingbot.service 