#!/bin/bash

# while true; do
#   if [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
#     echo "GPU 0 is free"
#     break
#   fi
#   echo "GPU 0 is busy, waiting 60s..."
#   sleep 60
# done

/bin/bash 0.profile_llama3.2_3b.sh
/bin/bash 0.profile_llama3.2_3b_ours.sh

