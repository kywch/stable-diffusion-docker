import io, base64, time

from flask import Flask, request, json, jsonify
from flask_cors import CORS
from waitress import serve
from PIL import Image

import torch
from torch import autocast

from visual_grounding import get_ofa_visual_grounding, create_mask
from inpainting import StableDiffusionInpaintingPipeline


device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

def image2str(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return str(base64.b64encode(buffered.getvalue()).decode("utf-8"))

def str2image(encstr):
    return Image.open(io.BytesIO(base64.b64decode(encstr)))


@app.route('/magic', methods=['POST'])
def magic():
    # original function: get_stable_diffusion_images
    # required parameters: init_image, mask_prompt, target_prompt
    # optional parameters: strength=0.75, guidance_scale=7.5, random_seed = 0, num_samples=1, n_iter=1
    
    # getting parameters from JSON
    # see https://github.com/richstokes/gpt2-api/blob/main/gpt2-api.py
    input_json = request.get_json(force=True)
    
    # check for valid input
    if request.method == 'POST':
        if input_json.get('init_image') and input_json.get('mask_prompt') and input_json.get('target_prompt'):

            # image is provided as base64 image
            init_image = str2image(input_json.get('init_image')).resize((512, 512))
            # required parameters: mask_prompt, target_prompt
            mask_prompt = input_json.get('mask_prompt')
            target_prompt = input_json.get('target_prompt')
            
            # optional parameters
            strength = float(input_json.get("strength")) if input_json.get("strength") else 0.75
            guidance_scale = float(input_json.get("guidance_scale")) if input_json.get("guidance_scale") else 7.5
            random_seed = float(input_json.get("random_seed")) if input_json.get("random_seed") else 0
            num_samples = int(input_json.get("num_samples")) if input_json.get("num_samples") else 1
            n_iter = int(input_json.get("n_iter")) if input_json.get("n_iter") else 1
            
            start_time = time.perf_counter()
            
            # create the mask
            coords, img = get_ofa_visual_grounding(init_image, mask_prompt)  #pil image as input
            mask_image = create_mask(coords, img)            

            # diffuse some magic
            encoded_images = []
            generator = torch.Generator(device=device).manual_seed(random_seed)
            for _ in range(n_iter):
                with autocast(device):
                    image_set = pipe(
                        prompt=[target_prompt] * num_samples,
                        init_image=init_image,
                        mask_image=mask_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )["sample"]

                for image in image_set:
                    encoded_images.append(image2str(image))
            
            return (jsonify({'result': encoded_images, 'execution_time': time.perf_counter() - start_time}), 200)
            
        else:
            return(jsonify({'error': 'Please pass required arguments'}), 404)
    else:
        return(jsonify({'error': 'Please POST to this endpoint'}), 404)            
    

def main():
    print('Please put the hf auth token to use stable diffusion...')
    token = input('token: ')
    global pipe
    pipe = StableDiffusionInpaintingPipeline.from_pretrained(
        model_path,
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=token
    ).to(device)
    serve(app, port=6000, threads=4)
    
if __name__ == "__main__":
    main()
    