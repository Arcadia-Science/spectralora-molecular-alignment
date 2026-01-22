#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${SRC_DIR:-/mnt/data/processed_all}"
S3_BUCKET="${S3_BUCKET:-molecule-dft-imputed}"
S3_PREFIX="${S3_PREFIX:-processed_all}"
AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_REGION:-}"

aws_args=()
if [[ -n "$AWS_PROFILE" ]]; then
  aws_args+=(--profile "$AWS_PROFILE")
fi
if [[ -n "$AWS_REGION" ]]; then
  aws_args+=(--region "$AWS_REGION")
fi

aws s3 sync "$SRC_DIR" "s3://$S3_BUCKET/$S3_PREFIX/" --only-show-errors "${aws_args[@]}"
