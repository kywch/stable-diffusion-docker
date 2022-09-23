#!/bin/bash
# Build n Run helper script
set -xe
docker build . -t fsdl-sdprj
docker run --rm --name fsdl-sdprj --cpus=2 -p 6000:6000 --gpus '"device=1"' -it fsdl-sdprj