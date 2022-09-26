#!/bin/bash
# Build n Run helper script
set -xe
docker build . -t fsdl-sdprj
docker run --rm -p 127.0.0.1:64000:6000 --gpus '"device=1"' -it fsdl-sdprj
