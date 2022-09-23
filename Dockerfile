FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# set -x prints commands and set -e causes us to stop on errors
RUN set -ex

# OFA install needs these & OFA's RefcocoTask (from pycocotools) requires gcc 
ARG APT_PACKAGES="git wget build-essential"

WORKDIR /app

# Install OFA by following https://colab.research.google.com/drive/1AHQNRdaUpRTgr3XySHSlba8aXwBAjwPB?usp=sharing
# also download and copy the pretrained OFA model file (ofa_large_384.pt)
RUN if [ -n "${APT_PACKAGES}" ]; then apt-get update && apt-get install --no-install-recommends -y ${APT_PACKAGES}; fi && \
    git clone --depth=1 https://github.com/OFA-Sys/OFA.git && \
    mkdir -p OFA/checkpoints/ && wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/ofa_large_384.pt && \
    mv ofa_large_384.pt OFA/checkpoints/ofa_large.pt && \
    cd OFA && sed '1d' requirements.txt | xargs -I {} pip install {} && cd - && \
    # now remove apt packages
    if [ -n "${APT_PACKAGES}" ]; then apt-get remove -y --auto-remove ${APT_PACKAGES} && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*; fi

# Install transformers and diffusion
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# OFA's Refcoco_infer runs within OFA, directly calling utils, tasks, models in the dir
WORKDIR /app/OFA

# Copy the work horses
COPY visual_grounding.py visual_grounding.py
COPY inpainting.py inpainting.py

# Copy the testing script
COPY app.py app.py

CMD python3 app.py