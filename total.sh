#!/bin/bash

while true; do
  if [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
    echo "GPU 0 is free"
    break
  fi
  echo "GPU 0 is busy, waiting 60s..."
  sleep 60
done

/bin/bash 0.profile_mistral.sh
/bin/bash 0.profile_vicuna.sh

sleep 18000
/bin/bash 1.eval_mistral.sh
/bin/bash 1.eval_vicuna.sh
