#!/bin/bash
# EC2 user data (Amazon Linux 2023, t3.medium). The instance profile must carry deploy/iam-policy.json.
# Set these three values before launching:
REPO_URL="https://github.com/YOUR-ORG/patient-followup-assistant.git"
S3_BUCKET="YOUR-FOLLOWUP-BUCKET"
SECRET_ID="followup/llm-keys"   # JSON secret: {"OPENAI_API_KEY": "...", "GOOGLE_API_KEY": "..."}
set -euo pipefail

dnf install -y docker git jq
systemctl enable --now docker
curl -sSL "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose --create-dirs
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

git clone "$REPO_URL" /opt/followup
cd /opt/followup

REGION=$(TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60") && \
  curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)

# API keys come from Secrets Manager at boot — never from the repository.
SECRET=$(aws secretsmanager get-secret-value --secret-id "$SECRET_ID" --region "$REGION" --query SecretString --output text)
cat > .env <<ENV
LLM_PROVIDER=auto
OPENAI_API_KEY=$(echo "$SECRET" | jq -r '.OPENAI_API_KEY // empty')
GOOGLE_API_KEY=$(echo "$SECRET" | jq -r '.GOOGLE_API_KEY // empty')
STORAGE_BACKEND=s3
S3_BUCKET=$S3_BUCKET
S3_PREFIX=followup
AWS_REGION=$REGION
VECTOR_BACKEND=chroma
ENV
chmod 600 .env

docker compose up -d --build
