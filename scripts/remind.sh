#!/usr/bin/env bash
# Periodic reminder script. Uses zero AI credits and runs entirely locally.

INTERVAL="${1:-120}"

while true; do
  echo "you got this! keep going"
  sleep "$INTERVAL"
done
