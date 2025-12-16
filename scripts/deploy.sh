#!/bin/bash
# Deployment script for manual deployment
# Can be run locally or on the server

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="rag-medicine-instructions"
APP_DIR="/opt/rag-medicine-instructions"
APP_USER="rag-app"

echo "=========================================="
echo "Deploying RAG Medicine Instructions"
echo "=========================================="

# Check if running on server
if [ ! -d "$APP_DIR" ]; then
    echo -e "${RED}Error: Application directory not found: $APP_DIR${NC}"
    echo "This script should be run on the server or with proper SSH access."
    exit 1
fi

echo ""
echo "Step 1: Checking service status..."
echo "----------------------------------------"

if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${YELLOW}Service is running. Stopping...${NC}"
    sudo systemctl stop "$SERVICE_NAME"
    echo -e "${GREEN}Service stopped${NC}"
else
    echo -e "${YELLOW}Service is not running${NC}"
fi

echo ""
echo "Step 2: Creating backup..."
echo "----------------------------------------"

BACKUP_DIR="$APP_DIR/backups"
BACKUP_NAME="backup-$(date +%Y%m%d-%H%M%S)"
sudo -u "$APP_USER" mkdir -p "$BACKUP_DIR"

# Backup ChromaDB
if [ -d "$APP_DIR/storage/chroma" ]; then
    echo "Backing up ChromaDB..."
    sudo -u "$APP_USER" tar -czf "$BACKUP_DIR/chromadb-$BACKUP_NAME.tar.gz" -C "$APP_DIR" storage/chroma
    echo -e "${GREEN}ChromaDB backed up${NC}"
fi

# Keep only last 5 backups
cd "$BACKUP_DIR"
sudo -u "$APP_USER" ls -t chromadb-*.tar.gz | tail -n +6 | xargs -r rm -f
echo -e "${GREEN}Old backups cleaned${NC}"

echo ""
echo "Step 3: Updating code..."
echo "----------------------------------------"

# This assumes code is already in APP_DIR (via git pull or rsync)
# In GitHub Actions, code will be deployed via rsync/git pull
echo -e "${YELLOW}Code should be updated via git pull or rsync before this step${NC}"

echo ""
echo "Step 4: Installing/updating dependencies..."
echo "----------------------------------------"

if [ -f "$APP_DIR/requirements.txt" ]; then
    echo "Installing Python dependencies..."
    sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
    sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
    echo -e "${GREEN}Dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

echo ""
echo "Step 5: Verifying .env file..."
echo "----------------------------------------"

if [ ! -f "$APP_DIR/.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Please create .env file with required environment variables"
else
    echo -e "${GREEN}.env file exists${NC}"
fi

echo ""
echo "Step 6: Starting service..."
echo "----------------------------------------"

sudo systemctl start "$SERVICE_NAME"

# Wait a moment for service to start
sleep 2

if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${GREEN}Service started successfully${NC}"
else
    echo -e "${RED}Error: Service failed to start${NC}"
    echo "Check logs with: sudo journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi

echo ""
echo "Step 7: Checking service status..."
echo "----------------------------------------"

sudo systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "=========================================="
echo -e "${GREEN}Deployment completed!${NC}"
echo "=========================================="
echo ""
echo "Service logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "Service status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""

