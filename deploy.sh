#!/bin/bash

# 🚀 Newstratbot AWS EC2 Quick Deploy Script
# Run this on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "🚀 Starting Newstratbot deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please don't run this script as root. Run as ubuntu user."
    exit 1
fi

print_status "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y

print_status "📦 Installing dependencies..."
sudo apt install -y python3-pip python3-venv nginx git htop curl wget unzip

print_status "🐍 Setting up Python environment..."
sudo mkdir -p /opt/newstratbot
sudo chown ubuntu:ubuntu /opt/newstratbot
cd /opt/newstratbot

print_status "📥 Cloning repository..."
git clone https://github.com/blacula3000/Newstratbot.git .

print_status "🔧 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

print_status "📋 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

print_status "📁 Creating directories..."
mkdir -p logs
mkdir -p /opt/backups/newstratbot

print_status "⚙️ Setting up Nginx configuration..."
sudo tee /etc/nginx/sites-available/newstratbot > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
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
EOF

sudo ln -sf /etc/nginx/sites-available/newstratbot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

print_status "🔧 Creating Gunicorn configuration..."
tee gunicorn.conf.py > /dev/null <<EOF
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
EOF

print_status "🚀 Creating systemd services..."
sudo tee /etc/systemd/system/newstratbot-web.service > /dev/null <<EOF
[Unit]
Description=Newstratbot Web Interface
After=network.target

[Service]
Type=notify
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/newstratbot
Environment=PATH=/opt/newstratbot/venv/bin
EnvironmentFile=-/opt/newstratbot/.env
ExecStart=/opt/newstratbot/venv/bin/gunicorn --config gunicorn.conf.py wsgi:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/newstratbot-trader.service > /dev/null <<EOF
[Unit]
Description=Newstratbot Trading Engine
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/newstratbot
Environment=PATH=/opt/newstratbot/venv/bin
EnvironmentFile=-/opt/newstratbot/.env
ExecStart=/opt/newstratbot/venv/bin/python trading_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_status "📝 Creating environment template..."
tee .env.template > /dev/null <<EOF
# 🔑 Trading API Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=True

BYBIT_API_KEY=your_bybit_api_key
BYBIT_SECRET_KEY=your_bybit_secret_key
BYBIT_TESTNET=True

ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# 🤖 Claude AI Configuration
CLAUDE_API_KEY=your_claude_api_key

# 🌐 Flask Configuration
FLASK_ENV=production
SECRET_KEY=$(openssl rand -base64 32)
DEBUG=False

# 📊 Trading Configuration
DEFAULT_POSITION_SIZE=1000
MAX_POSITIONS=5
RISK_PERCENTAGE=2
STOP_LOSS_PERCENTAGE=2
TAKE_PROFIT_PERCENTAGE=4

# 📈 STRAT Agent Configuration
AGENT_UPDATE_INTERVAL=3600
ENABLE_REAL_TIME_ANALYSIS=True
LOG_LEVEL=INFO
EOF

print_status "🛡️ Setting up firewall..."
sudo ufw --force enable
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

print_status "🔧 Configuring services..."
sudo systemctl daemon-reload
sudo systemctl enable nginx newstratbot-web
sudo nginx -t

print_status "📋 Creating backup script..."
tee backup.sh > /dev/null <<'EOF'
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

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "✅ Backup completed: $DATE"
EOF

chmod +x backup.sh

print_status "⚙️ Setting up log rotation..."
sudo tee /etc/logrotate.d/newstratbot > /dev/null <<EOF
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
EOF

print_success "🎉 Deployment completed successfully!"

echo ""
echo "📋 Next Steps:"
echo "1. 📝 Edit environment file: nano /opt/newstratbot/.env"
echo "2. 🔑 Add your API keys (copy from .env.template)"
echo "3. 🚀 Start services: sudo systemctl start nginx newstratbot-web"
echo "4. 🌐 Access dashboard: http://$(curl -s ifconfig.me)"
echo "5. 📊 Check logs: tail -f /opt/newstratbot/logs/gunicorn.log"
echo ""
echo "🤖 STRAT Agents available:"
find .claude/agents -name "*.md" -exec basename {} .md \; | sort

echo ""
print_warning "⚠️  Remember to configure your API keys before starting the trading services!"
print_success "🚀 Your Newstratbot is ready to deploy!"