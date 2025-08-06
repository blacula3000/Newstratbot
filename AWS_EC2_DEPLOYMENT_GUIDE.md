# 🚀 AWS EC2 Deployment Guide - Newstratbot STRAT Agent System

## 📋 **Prerequisites Checklist**

### 🔑 **Required Accounts & API Keys:**
- [ ] AWS Account with EC2 access
- [ ] Trading platform API keys (Binance, Bybit, Alpaca)
- [ ] Claude API key for agent system
- [ ] Domain name (optional for HTTPS)

### 💻 **Local Requirements:**
- [ ] AWS CLI installed and configured
- [ ] SSH client (Terminal/PuTTY)
- [ ] Git installed locally

---

## 🏗️ **Phase 1: AWS EC2 Instance Setup**

### 🖥️ **1.1 Launch EC2 Instance**

1. **Login to AWS Console** 🔐
   - Navigate to EC2 Dashboard
   - Click "Launch Instance"

2. **Instance Configuration** ⚙️
   ```
   📍 Name: Newstratbot-Production
   🖥️ AMI: Ubuntu Server 22.04 LTS (Free Tier Eligible)
   💾 Instance Type: t3.medium (recommended) or t2.micro (testing)
   🔑 Key Pair: Create new key pair "newstratbot-key"
   ```

3. **Security Group Setup** 🛡️
   ```
   Group Name: newstratbot-sg
   
   Inbound Rules:
   - SSH (22): Your IP only
   - HTTP (80): 0.0.0.0/0 (for web interface)
   - HTTPS (443): 0.0.0.0/0 (if using SSL)
   - Custom TCP (5000): Your IP only (Flask dev server)
   ```

4. **Storage Configuration** 💾
   ```
   Root Volume: 20 GB gp3 SSD (minimum)
   ```

5. **Launch Instance** 🚀
   - Download the `.pem` key file
   - Store securely (you'll need this for SSH access)

### 🔧 **1.2 Connect to Instance**

**Windows (PowerShell):**
```powershell
# Set key permissions
icacls "newstratbot-key.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"

# Connect via SSH
ssh -i "newstratbot-key.pem" ubuntu@YOUR_INSTANCE_PUBLIC_IP
```

**Mac/Linux:**
```bash
# Set key permissions
chmod 400 newstratbot-key.pem

# Connect via SSH
ssh -i newstratbot-key.pem ubuntu@YOUR_INSTANCE_PUBLIC_IP
```

---

## ⚡ **Phase 2: Server Environment Setup**

### 🔄 **2.1 System Updates & Dependencies**

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-pip python3-venv nginx git htop curl wget unzip

# Install Node.js (for potential frontend enhancements)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Docker (optional for containerization)
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
```

### 🐍 **2.2 Python Environment Setup**

```bash
# Create application directory
sudo mkdir -p /opt/newstratbot
sudo chown ubuntu:ubuntu /opt/newstratbot
cd /opt/newstratbot

# Clone the repository
git clone https://github.com/blacula3000/Newstratbot.git .

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional packages for production
pip install gunicorn supervisor
```

### 🔐 **2.3 Environment Variables Configuration**

```bash
# Create environment file
sudo nano /opt/newstratbot/.env
```

**Add the following content:**
```env
# 🔑 Trading API Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=True  # Set to False for live trading

BYBIT_API_KEY=your_bybit_api_key
BYBIT_SECRET_KEY=your_bybit_secret_key
BYBIT_TESTNET=True    # Set to False for live trading

ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use https://api.alpaca.markets for live

# 🤖 Claude AI Configuration
CLAUDE_API_KEY=your_claude_api_key

# 🌐 Flask Configuration
FLASK_ENV=production
SECRET_KEY=your_super_secret_key_here_change_this
DEBUG=False

# 📊 Trading Configuration
DEFAULT_POSITION_SIZE=1000
MAX_POSITIONS=5
RISK_PERCENTAGE=2
STOP_LOSS_PERCENTAGE=2
TAKE_PROFIT_PERCENTAGE=4

# 📈 STRAT Agent Configuration
AGENT_UPDATE_INTERVAL=3600  # 1 hour in seconds
ENABLE_REAL_TIME_ANALYSIS=True
LOG_LEVEL=INFO
```

**Secure the environment file:**
```bash
sudo chmod 600 /opt/newstratbot/.env
```

---

## 🌐 **Phase 3: Web Server Configuration**

### 🔧 **3.1 Nginx Configuration**

```bash
# Create Nginx configuration
sudo nano /etc/nginx/sites-available/newstratbot
```

**Add configuration:**
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 30;
        proxy_send_timeout 30;
        proxy_read_timeout 30;
    }

    location /static {
        alias /opt/newstratbot/static;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
}
```

**Enable the site:**
```bash
# Enable site and remove default
sudo ln -s /etc/nginx/sites-available/newstratbot /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default

# Test and reload Nginx
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl enable nginx
```

### 🔐 **3.2 SSL Certificate (Optional but Recommended)**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d YOUR_DOMAIN

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 🚀 **Phase 4: Application Deployment**

### 📋 **4.1 Gunicorn Configuration**

```bash
# Create Gunicorn configuration
sudo nano /opt/newstratbot/gunicorn.conf.py
```

**Add configuration:**
```python
import multiprocessing

bind = "127.0.0.1:5000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
timeout = 30
keepalive = 2
user = "ubuntu"
group = "ubuntu"
tmp_upload_dir = None
logfile = "/opt/newstratbot/logs/gunicorn.log"
loglevel = "info"
accesslog = "/opt/newstratbot/logs/access.log"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
```

### 📁 **4.2 Create Required Directories**

```bash
# Create log directory
mkdir -p /opt/newstratbot/logs

# Set permissions
sudo chown -R ubuntu:ubuntu /opt/newstratbot
chmod +x /opt/newstratbot/*.sh
```

### ⚙️ **4.3 Systemd Service Configuration**

```bash
# Create systemd service for the web interface
sudo nano /etc/systemd/system/newstratbot-web.service
```

**Add service configuration:**
```ini
[Unit]
Description=Newstratbot Web Interface
After=network.target

[Service]
Type=notify
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/newstratbot
Environment=PATH=/opt/newstratbot/venv/bin
EnvironmentFile=/opt/newstratbot/.env
ExecStart=/opt/newstratbot/venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Create systemd service for the trading bot
sudo nano /etc/systemd/system/newstratbot-trader.service
```

```ini
[Unit]
Description=Newstratbot Trading Engine
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/newstratbot
Environment=PATH=/opt/newstratbot/venv/bin
EnvironmentFile=/opt/newstratbot/.env
ExecStart=/opt/newstratbot/venv/bin/python trading_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start services:**
```bash
# Reload systemd and enable services
sudo systemctl daemon-reload
sudo systemctl enable newstratbot-web newstratbot-trader
sudo systemctl start newstratbot-web newstratbot-trader

# Check status
sudo systemctl status newstratbot-web
sudo systemctl status newstratbot-trader
```

---

## 🧪 **Phase 5: Testing & Validation**

### ✅ **5.1 System Health Checks**

```bash
# Check if services are running
sudo systemctl status newstratbot-web
sudo systemctl status newstratbot-trader

# Check ports
sudo netstat -tlnp | grep :5000
sudo netstat -tlnp | grep :80

# Check logs
tail -f /opt/newstratbot/logs/gunicorn.log
tail -f /opt/newstratbot/logs/trading_bot.log
```

### 🌐 **5.2 Web Interface Testing**

1. **Open browser and navigate to:**
   ```
   http://YOUR_INSTANCE_PUBLIC_IP
   # or
   https://YOUR_DOMAIN (if SSL configured)
   ```

2. **Test Features:**
   - [ ] 📊 Dashboard loads correctly
   - [ ] 🔄 Real-time data updates
   - [ ] 📈 STRAT agent reports display
   - [ ] ⚙️ Configuration panel works
   - [ ] 📋 Trading logs visible

### 🤖 **5.3 STRAT Agent System Testing**

```bash
# Test Claude API connection
cd /opt/newstratbot
source venv/bin/activate
python -c "
import os
from claude_api import test_connection
print('🤖 Testing Claude API connection...')
result = test_connection(os.getenv('CLAUDE_API_KEY'))
print(f'✅ Connection: {result}')
"

# Test trading API connections
python test_setup.py

# Check agent system
python -c "
import glob
agents = glob.glob('.claude/agents/*.md')
print(f'🎯 Found {len(agents)} STRAT agents:')
for agent in agents:
    print(f'  - {agent.split(\"/\")[-1]}')
"
```

### 📈 **5.4 Performance Testing**

```bash
# Install monitoring tools
sudo apt install -y htop iotop nethogs

# Monitor system resources
htop

# Check memory usage
free -h

# Check disk space
df -h

# Monitor network usage
sudo nethogs
```

---

## 📊 **Phase 6: Monitoring & Maintenance**

### 📈 **6.1 CloudWatch Integration (Optional)**

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure CloudWatch
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### 🔄 **6.2 Automated Backups**

```bash
# Create backup script
sudo nano /opt/newstratbot/backup.sh
```

```bash
#!/bin/bash
# 💾 Newstratbot Backup Script

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/newstratbot"
mkdir -p $BACKUP_DIR

# Backup application files
tar -czf $BACKUP_DIR/newstratbot_$DATE.tar.gz \
    --exclude='venv' \
    --exclude='logs/*.log' \
    --exclude='__pycache__' \
    /opt/newstratbot/

# Backup database (if using SQLite)
if [ -f /opt/newstratbot/trading_data.db ]; then
    cp /opt/newstratbot/trading_data.db $BACKUP_DIR/database_$DATE.db
fi

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
find $BACKUP_DIR -name "*.db" -mtime +7 -delete

echo "✅ Backup completed: $DATE"
```

```bash
# Make executable and schedule
chmod +x /opt/newstratbot/backup.sh
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/newstratbot/backup.sh") | crontab -
```

### 🔍 **6.3 Log Rotation**

```bash
# Configure log rotation
sudo nano /etc/logrotate.d/newstratbot
```

```
/opt/newstratbot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 ubuntu ubuntu
    postrotate
        systemctl reload newstratbot-web
    endscript
}
```

---

## 🚨 **Phase 7: Security Hardening**

### 🔐 **7.1 Firewall Configuration**

```bash
# Enable UFW firewall
sudo ufw enable

# Allow specific ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS

# Check status
sudo ufw status verbose
```

### 🛡️ **7.2 SSH Hardening**

```bash
# Edit SSH configuration
sudo nano /etc/ssh/sshd_config
```

**Recommended changes:**
```
Port 22
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
```

```bash
# Restart SSH service
sudo systemctl restart ssh
```

### 🔒 **7.3 API Key Security**

```bash
# Ensure environment file is secure
sudo chmod 600 /opt/newstratbot/.env
sudo chown ubuntu:ubuntu /opt/newstratbot/.env

# Check for exposed keys in logs
sudo grep -r "api" /opt/newstratbot/logs/ | head -10
```

---

## 🎯 **Phase 8: Final Testing Checklist**

### ✅ **Complete System Test**

- [ ] 🌐 **Web Interface**: Accessible via public IP/domain
- [ ] 🤖 **STRAT Agents**: All 10 agents loading correctly
- [ ] 📊 **Data Feeds**: Real-time market data flowing
- [ ] 💹 **Trading APIs**: Connection to exchanges working
- [ ] 🔄 **Agent Reports**: Hourly updates generating
- [ ] 📈 **Dashboard**: Real-time updates displaying
- [ ] 🔐 **Security**: SSL working, firewall active
- [ ] 📋 **Logging**: All logs writing properly
- [ ] ⚡ **Performance**: Response times < 2 seconds
- [ ] 🔄 **Auto-restart**: Services restart after reboot

### 📱 **Access Information**

**🌐 Web Dashboard:**
```
URL: http://YOUR_INSTANCE_PUBLIC_IP
# or https://YOUR_DOMAIN

Default Login: Check web_interface.py for authentication
```

**🔧 System Management:**
```bash
# Start/Stop Services
sudo systemctl start/stop newstratbot-web
sudo systemctl start/stop newstratbot-trader

# View Logs
tail -f /opt/newstratbot/logs/trading_bot.log
tail -f /opt/newstratbot/logs/gunicorn.log

# Update Application
cd /opt/newstratbot
git pull origin main
sudo systemctl restart newstratbot-web newstratbot-trader
```

---

## 🚀 **Quick Deployment Commands**

**One-liner setup (run on fresh EC2 instance):**
```bash
wget -O - https://raw.githubusercontent.com/blacula3000/Newstratbot/main/deploy.sh | bash
```

**Manual deployment summary:**
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y && sudo apt install -y python3-pip python3-venv nginx git

# 2. Clone and setup
cd /opt && sudo git clone https://github.com/blacula3000/Newstratbot.git newstratbot
cd newstratbot && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 3. Configure environment
sudo nano .env  # Add your API keys

# 4. Start services
sudo systemctl enable --now newstratbot-web newstratbot-trader nginx
```

---

## 📞 **Support & Troubleshooting**

### 🆘 **Common Issues:**

1. **Service won't start:**
   ```bash
   sudo journalctl -u newstratbot-web -f
   sudo journalctl -u newstratbot-trader -f
   ```

2. **Web interface not accessible:**
   ```bash
   sudo nginx -t
   sudo systemctl status nginx
   curl -I http://localhost:5000
   ```

3. **Agent system not working:**
   ```bash
   source venv/bin/activate
   python -c "import claude_api; print('Claude API available')"
   ```

4. **High memory usage:**
   ```bash
   sudo systemctl edit newstratbot-trader
   # Add: [Service] Environment="PYTHONUNBUFFERED=1"
   ```

### 🔧 **Performance Optimization:**
- Increase EC2 instance size if needed
- Enable swap if memory constrained
- Use Redis for caching (optional)
- Enable gzip compression in Nginx

**🎉 Your Newstratbot with full STRAT agent ecosystem is now ready for production deployment!** 🚀📈
