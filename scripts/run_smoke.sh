#!/usr/bin/env bash
set -euo pipefail
distill-safe pipeline run --config configs/pipelines/smoke_test.yaml
