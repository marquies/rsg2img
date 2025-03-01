import sys
import os
import uuid

import shutil # save img locally
import pyyed
import requests # request img from web
from openai import OpenAI

from io import BytesIO
import IPython
import json
import os
import base64
from PIL import Image
import requests
import time
from random import randint

import sgconversion as converter

# @markdown To get your API key visit https://platform.stability.ai/account/keys
STABILITY_KEY = ""

def strip_noun(input, words):
    for word in words:
        if word in input.lower():
            return word
    return input

def send_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def generate_image_local(prompt, filename, output_path):
    # send rest post with json
    prompt = prompt
    aspect_ratio = "1:1" #@param ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]

    seed = randint(0, 100000)
    output_format = "jpeg" #@param ["jpeg", "png"]
    
    host = f"http://cudaknecht:7860/sdapi/v1/txt2img"

    params = {
        "prompt" : prompt,
        #"
        #"

        "aspect_ratio" : "16:9",
        "seed" : seed,
        "output_format" : "jpeg",
        "width": 1920,
        "height": 1080,
        "steps": 150
    }


    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        json=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    base64_image_data = response.json()['images'][0]

    # Decode the base64 encoded image data
    image_data = base64.b64decode(base64_image_data)

    # Save and display result
    generated = f"{output_path}/generated_{filename}_{seed}.{output_format}"
    prompt_txt = f"{output_path}/prompt_{filename}_{seed}.txt"

    # Specify the path where you want to save the image
    with open(generated, 'wb') as file:
        file.write(image_data)


    image = Image.open(generated)
    new_image = image.resize((1920, 1080))
    new_image.save(generated)
    # Write the decoded image data to a file
    
    print(f"Saved image {generated}")
    
    #output.no_vertical_scroll()
    print("Result image:")
    #IPython.display.display(Image.open(generated))
    
    
    with open(prompt_txt, "w") as f:
        f.write(prompt)


#    curl -X 'POST' \
#  'http://cudaknecht:7860/sdapi/v1/txt2img' \
#  -H 'accept: application/json' \
#  -H 'Content-Type: application/json' \
#  -d '{
#  "prompt": "A parking lot in a city scape. Some cars are parking in the parking lot. Left is a small park with trees. Right is a  thin 10 level building, reading Pizza on it.",
#  "negative_prompt": "",
#  "styles": [
#    "string"
#  ],
#  "seed": -1,
#  "subseed": -1,
#  "subseed_strength": 0,
#  "seed_resize_from_h": -1,
#  "seed_resize_from_w": -1,
#  "sampler_name": "string",
#  "scheduler": "string",
#  "batch_size": 1,
#  "n_iter": 1,
#  "steps": 150,
#  "cfg_scale": 7,
#  "width": 512,
#  "height": 512,
#  "restore_faces": true,
#  "tiling": true,
#  "do_not_save_samples": false,
#  "do_not_save_grid": false,
#  "eta": 0,
#  "denoising_strength": 0,
#  "s_min_uncond": 0,
#  "s_churn": 0,
#  "s_tmax": 0,
#  "s_tmin": 0,
#  "s_noise": 0,
#  "override_settings": {},
#  "override_settings_restore_afterwards": true,
#  "refiner_checkpoint": "sd3_medium.safetensors",
#  "refiner_switch_at": 0,
#  "disable_extra_networks": false,
#  "firstpass_image": "string",
#  "comments": {},
#  "enable_hr": false,
#  "firstphase_width": 0,
#  "firstphase_height": 0,
#  "hr_scale": 2,
#  "hr_upscaler": "string",
#  "hr_second_pass_steps": 0,
#  "hr_resize_x": 0,
#  "hr_resize_y": 0,
#  "hr_checkpoint_name": "sd3_medium.safetensors",
#  "hr_sampler_name": "string",
#  "hr_scheduler": "string",
#  "hr_prompt": "",
#  "hr_negative_prompt": "",
#  "force_task_id": "string",
#  "sampler_index": "Euler",
#  "script_args": [],
#  "send_images": true,
#  "save_images": false,
#  "alwayson_scripts": {}
#}'


def generate_image(prompt, filename, output_path):
    #prompt = "glowing mushroom in the alchemist's garden" #@param {type:"string"}
    aspect_ratio = "1:1" #@param ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
    seed = 0 #@param {type:"integer"}
    output_format = "jpeg" #@param ["jpeg", "png"]

    host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"
    
    params = {
        "prompt" : prompt,
        #"prompt" : "man besides tree. tree on grass. CGI",
        "aspect_ratio" : "16:9",
        "seed" : 0,
        "output_format" : "jpeg",
        #"model" : "sd3-medium"
        "model" : "sd3-large",
        "mode" : "text-to-image"
    }
    
    response = send_generation_request(
        host,
        params
    )
    
    # Decode response
    output_image = response.content
    finish_reason = response.headers.get("finish-reason")
    seed = response.headers.get("seed")
    
    # Check for NSFW classification
    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("Generation failed NSFW classifier")
    
    #resize output_image to 1920x1080



    # Save and display result
    generated = f"{output_path}/generated_{filename}_{seed}.{output_format}"
    prompt_txt = f"{output_path}/prompt_{filename}_{seed}.txt"
    with open(generated, "wb") as f:
        f.write(output_image)
    print(f"Saved image {generated}")

    image = Image.open(generated)
    new_image = image.resize((1920, 1080))
    new_image.save(generated)
    
    #output.no_vertical_scroll()
    print("Result image:")
    #IPython.display.display(Image.open(generated))
    
    
    with open(prompt_txt, "w") as f:
        f.write(prompt)

# main function
def main():
    # Create the output directory
    output_path = "output"
    #os.makedirs(output_path, exist_ok=True)
    
    #txt_files = ["output/recording66/frame_475.txt"]
    txt_files = []
    #get all txt files in output directory and subdirectories
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.startswith('frame') and file.endswith('.txt'):
                #dir = root.split('/')[-1]
                
                filename = file.split('.')[0]
                checkname = "generated_" + filename

                #check if foot contains a file, which name start with checkname
                if not any(checkname in f for f in os.listdir(root)):
                    txt_files.append(os.path.join(root, file))
                else:
                    print(f"Skipping {os.path.join(root,file)} as it has already been processed.")
                    
                
    client = OpenAI()
    for txt_file in txt_files:
        # Load the scene graph
        inputs, camera = converter.filter_log_file(txt_file, "Main Camera")
        print("Found " + str(len(inputs)) + " objects in the log file.  ")

        # Preprocess
        objects = converter.get_objects_from_input(inputs)
        #print("Filtered " + str(len(objects)) + " objects from the log file.   ")

        # Add semantics
        tuples = converter.determine_arrangement2(camera, objects)
        tuples = converter.remove_bidirectional_duplicates(tuples)

        # Postprocess
        my_file = open("nouns.txt", "r") 
        data = my_file.read() 
        data_into_list = data.replace('\n', '#').split("#") 
        #print(data_into_list) 
        my_file.close() 

        # 1. convert object names
        # 2. filter double entries
        processed_pairs = set()

        prompt = "" # pylint: disable=invalid-name
        for element in tuples:
            objA = element.x.split('/')[-1]
            objB = element.z.split('/')[-1]
            objA = strip_noun(objA, data_into_list)
            objB = strip_noun(objB, data_into_list)
            #print("\n" + objA + " is " + element.y + " " + objB + ".")
            #prompt += "\n" + objA + " is " + element.y + " " + objB + "."
            # Create a tuple of the pair
            pair = (objA, objB)
    
            # Check if the pair is already processed
            if pair not in processed_pairs:
                # Add the pair to the set of processed pairs
                processed_pairs.add(pair)
        
                # Append the prompt
                prompt += "\n" + objA + " is " + element.y + " " + objB + "."

        if prompt == "":
            prompt = "Empty scene"
        
        # put into simplify prompt
        nprompt = "I have a representation of a graph, defining the positions of objects in a 3d space. This results in many items. I want to use them in a prompt for AI image to text generation, but there are too many. Simplify descriptions by separating them into hierarchical levels: foreground, midground, and background, or by their relative importance. Just print the text. Here is the input: " + prompt

            
        cc=client.chat.completions.create(
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        #{"role": "assistant", "content": "relevant: farts exert force"}, # RAG
        {"role": "user", "content": nprompt}
        ], stream=True, max_tokens=420, top_p=.69, model="gpt-4o")
        
        #print(*(ck.choices[0].delta.content or "" for ck in cc), sep="", end="")
        result = "".join(ck.choices[0].delta.content or "" for ck in cc)

        prompt = result
        prompt += ". \n CGI"

        # get base path of txt file
        base_path = os.path.dirname(txt_file)
        # get filename of txt file without extension
        filename = os.path.splitext(os.path.basename(txt_file))[0]

        # Generate the image
        generate_image(prompt, filename, base_path)
        #generate_image_local(prompt, filename, base_path)

    # Load the scene graphs
    #scene_graphs = [

if __name__ == "__main__":
	main()
