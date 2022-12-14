# stable-diffusion-docker

Make it run with this docker backend: https://github.com/yvrjsharma/HugginFace_Gradio/blob/main/GradioApp_VisualGrounding_guided_Inpainting_StableDiffusion.ipynb



## Build n Run

`./bnr.sh`

Uses a docker image to remove the complexity of getting a working pytorch + OFA + stable diffusion environment working locally.

When running the docker, s/he should provide [the Hugging Face access token](https://huggingface.co/docs/hub/security-tokens) to download the pre-trained stable diffusion and run the app.

* build the docker image
`docker build --tag fsdl-sdprj .`

* Is GPU accessible? (requires nvidia-docker)
`docker run --gpus '"device=1"' fsdl-sdprj nvidia-smi`

* interactive shell: 
`docker run --gpus '"device=1"' -it fsdl-sdprj /bin/bash`


## To-dos
- [V] Run the docker in the lambda cluster
- [V] Expose the API via ngrok (refer https://blog.ngrok.com/posts/how-to-secure-your-network-tunnels-with-oauth-fast)
- [V] Create [a colab notebook that runs on the API](https://colab.research.google.com/github/kywch/stable-diffusion-docker/blob/main/inpaint_colab_with_api.ipynb)


## references
* Docker inspiration: https://github.com/richstokes/gpt2-api
* installing nvidia-docker (`sudo apt-get install -y nvidia-docker2`): https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit
* moving the docker directory (if your root partition is small): https://www.digitalocean.com/community/questions/how-to-move-the-default-var-lib-docker-to-another-directory-for-docker-on-linux
