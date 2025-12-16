#!/bin/bash
# Setup script for Google Cloud server
# Run this script on the server to set up the environment

set -e  # Exit on error

echo "=========================================="
echo "Setting up RAG Medicine Instructions Server"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: Running as root. Some steps may need sudo.${NC}"
fi

# Variables
APP_USER="rag-app"
APP_DIR="/opt/rag-medicine-instructions"
PYTHON_VERSION="3.11"

echo ""
echo "Step 1: Installing system dependencies..."
echo "----------------------------------------"

# Update package list
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx
elif command -v yum &> /dev/null; then
    sudo yum install -y python3 python3-pip git nginx certbot python3-certbot-nginx
else
    echo -e "${RED}Error: Unsupported package manager. Please install Python 3.11+, git, nginx, and certbot manually.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}Python version: $(python3 --version)${NC}"

echo ""
echo "Step 2: Creating application user..."
echo "----------------------------------------"

# Create user if doesn't exist
if ! id "$APP_USER" &>/dev/null; then
    sudo useradd -r -m -s /bin/bash "$APP_USER"
    echo -e "${GREEN}Created user: $APP_USER${NC}"
else
    echo -e "${YELLOW}User $APP_USER already exists${NC}"
fi

echo ""
echo "Step 3: Creating application directory..."
echo "----------------------------------------"

sudo mkdir -p "$APP_DIR"
sudo chown "$APP_USER:$APP_USER" "$APP_DIR"
echo -e "${GREEN}Created directory: $APP_DIR${NC}"

echo ""
echo "Step 4: Creating data directories..."
echo "----------------------------------------"

sudo -u "$APP_USER" mkdir -p "$APP_DIR/data/html"
sudo -u "$APP_USER" mkdir -p "$APP_DIR/data/mht"
sudo -u "$APP_USER" mkdir -p "$APP_DIR/storage/chroma"
echo -e "${GREEN}Created data directories${NC}"

echo ""
echo "Step 5: Setting up Python virtual environment..."
echo "----------------------------------------"

# Switch to app user and create venv
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
echo -e "${GREEN}Created virtual environment${NC}"

echo ""
echo "Step 6: Installing Python dependencies..."
echo "----------------------------------------"

# Note: requirements.txt will be installed during deployment
# This is just a placeholder
echo -e "${YELLOW}Python dependencies will be installed during deployment${NC}"

echo ""
echo "Step 7: Setting up log directory..."
echo "----------------------------------------"

sudo mkdir -p /var/log/rag-medicine-instructions
sudo chown "$APP_USER:$APP_USER" /var/log/rag-medicine-instructions
echo -e "${GREEN}Created log directory${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}Server setup completed!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy the application code to $APP_DIR"
echo "2. Create .env file with required environment variables"
echo "3. Install systemd service file"
echo "4. Configure nginx"
echo "5. Set up SSL certificate with certbot"
echo ""
echo "To connect to the server:"
echo "  gcloud compute ssh rag-medicine-instructions-01 --zone us-east1-c"
echo ""

