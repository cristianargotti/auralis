#!/usr/bin/env bash
# ── AURALIS Deploy to EC2 ──
# Usage: ./scripts/deploy.sh
set -euo pipefail

# ── Config ──
AWS_PROFILE="mibaggy-co"
AWS_REGION="us-east-1"
INSTANCE_TYPE="t3.small"
AMI_ID="ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS us-east-1
KEY_NAME="meetmind"
KEY_FILE="$HOME/.ssh/meetmind.pem"
INSTANCE_NAME="auralis"
SG_NAME="auralis-sg"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export AWS_PROFILE AWS_REGION

echo "🎛️  AURALIS Deploy"
echo "   Instance: $INSTANCE_TYPE"
echo "   Region:   $AWS_REGION"
echo ""

# ── Step 1: Security Group ──
echo "🔒 Setting up security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "AURALIS - AI Music Production Engine" \
        --query 'GroupId' --output text)

    MY_IP=$(curl -s https://checkip.amazonaws.com)/32

    # SSH from your IP
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr "$MY_IP" > /dev/null

    # HTTP/HTTPS from anywhere
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 80 --cidr "0.0.0.0/0" > /dev/null
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
        --protocol tcp --port 443 --cidr "0.0.0.0/0" > /dev/null

    echo "   Created SG: $SG_ID (SSH from $MY_IP)"
else
    echo "   Using existing SG: $SG_ID"
fi

# ── Step 2: Check for existing instance ──
echo "🔍 Checking for existing AURALIS instance..."
EXISTING_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=running,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo "None")

if [ "$EXISTING_ID" != "None" ] && [ -n "$EXISTING_ID" ]; then
    INSTANCE_ID="$EXISTING_ID"
    STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)

    if [ "$STATE" = "stopped" ]; then
        echo "   Starting stopped instance $INSTANCE_ID..."
        aws ec2 start-instances --instance-ids "$INSTANCE_ID" > /dev/null
        aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    fi
    echo "   Using existing instance: $INSTANCE_ID"
else
    # ── Step 3: Launch new instance ──
    echo "🚀 Launching EC2 instance ($INSTANCE_TYPE)..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --associate-public-ip-address \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --query 'Instances[0].InstanceId' --output text)

    echo "   Instance: $INSTANCE_ID"
    echo "   Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
fi

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "   Public IP: $PUBLIC_IP"

# ── Step 4: Wait for SSH ──
echo "⏳ Waiting for SSH..."
for i in $(seq 1 30); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" 'echo ok' 2>/dev/null; then
        break
    fi
    sleep 5
done

# ── Step 5: Install Docker on instance ──
echo "🐳 Installing Docker..."
ssh -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" 'bash -s' << 'INSTALL_DOCKER'
if ! command -v docker &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker ubuntu
    echo "Docker installed ✓"
else
    echo "Docker already installed ✓"
fi
INSTALL_DOCKER

# ── Step 6: Sync project files ──
echo "📦 Syncing project files..."
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'node_modules' \
    --exclude '.next' \
    --exclude '__pycache__' \
    --exclude '.mypy_cache' \
    --exclude '.ruff_cache' \
    --exclude '.pytest_cache' \
    --exclude '.coverage' \
    -e "ssh -i $KEY_FILE" \
    "$PROJECT_DIR/" ubuntu@"$PUBLIC_IP":~/auralis/

# ── Step 7: Build and start ──
echo "🏗️  Building and starting containers..."
ssh -i "$KEY_FILE" ubuntu@"$PUBLIC_IP" 'bash -s' << 'START_APP'
cd ~/auralis
# Need to use sg docker for first run without logout/login
sg docker -c "docker compose up -d --build"
sg docker -c "docker compose ps"
START_APP

echo ""
echo "============================================================"
echo "🎛️  AURALIS is LIVE!"
echo "============================================================"
echo ""
echo "   URL:         http://$PUBLIC_IP"
echo "   API Docs:    http://$PUBLIC_IP/docs"
echo "   Health:      http://$PUBLIC_IP/health"
echo "   Instance:    $INSTANCE_ID"
echo ""
echo "   SSH:         ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo ""
echo "   Stop:        aws ec2 stop-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE"
echo "   Start:       aws ec2 start-instances --instance-ids $INSTANCE_ID --profile $AWS_PROFILE"
echo "============================================================"
