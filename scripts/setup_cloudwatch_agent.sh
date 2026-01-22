#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scripts/ec2_env.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

LOG_FILE="${LOG_FILE:-$OUTPUT_ROOT/run.log}"
AWS_REGION="${AWS_REGION:-}"
CW_LOG_GROUP="${CW_LOG_GROUP:-/detanet/pipeline}"
CW_LOG_STREAM="${CW_LOG_STREAM:-}"

if [[ -z "$AWS_REGION" ]]; then
  AWS_REGION="$(curl -s http://169.254.169.254/latest/meta-data/placement/region || true)"
fi
if [[ -z "$AWS_REGION" ]]; then
  echo "AWS_REGION is required."
  exit 1
fi

if [[ -z "$CW_LOG_STREAM" ]]; then
  CW_LOG_STREAM="$(curl -s http://169.254.169.254/latest/meta-data/instance-id || true)"
fi

if command -v dnf >/dev/null 2>&1; then
  sudo dnf -y install amazon-cloudwatch-agent
else
  sudo yum -y install amazon-cloudwatch-agent
fi

CONFIG_PATH="/opt/aws/amazon-cloudwatch-agent/bin/pipeline-log-config.json"
sudo bash -c "cat > ${CONFIG_PATH}" <<EOF
{
  "agent": {
    "region": "${AWS_REGION}"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "${LOG_FILE}",
            "log_group_name": "${CW_LOG_GROUP}",
            "log_stream_name": "${CW_LOG_STREAM}",
            "timestamp_format": "%Y-%m-%d %H:%M:%S"
          }
        ]
      }
    }
  }
}
EOF

sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -c file:${CONFIG_PATH} \
  -s

echo "CloudWatch agent configured for ${LOG_FILE} -> ${CW_LOG_GROUP}/${CW_LOG_STREAM}"
