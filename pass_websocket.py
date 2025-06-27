#This is an example that uses the websockets api to know when a prompt execution is done
#Once the prompt execution is done it downloads the images using the /history endpoint

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import sys
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io
import os

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

image_node_id = "18"
mask_node_id = "11"
prompt_node_id = "159"
original_image_node_id = "151"

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    #print(p)
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def upload_image(image_bytes, filename, image_type="input"):
    image = Image.open(image_bytes)
    temp_path = os.path.join("/tmp", filename)
    image.save(temp_path)
    return temp_path


def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            # If you want to be able to decode the binary stream for latent previews, here is how you can do it:
            # bytesIO = BytesIO(out[8:])
            # preview_image = Image.open(bytesIO) # This is your preview in PIL image format, store it in a global
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def save_images(images, output_path="./"):
    #Commented out code to display the output images:
    count = 0
    for node_id in images:
        for image_data in images[node_id]:
            count = count + 1
            filename = f"tempoutput{count}.png"
            image = Image.open(io.BytesIO(image_data))
            image.save(output_path + filename)

def run_pass(prompt_insertion, image_path, mask_path, original_image_path, workflow_path="./BuildingEdit.json"):
    #print(workflow_path)
    with open(workflow_path, 'r', encoding='utf-8') as file: 
      prompt_text = file.read()

    prompt = json.loads(prompt_text)

  
    #set the text prompt for our positive CLIPTextEncode
    prompt[prompt_node_id]["inputs"]["value"] = prompt_insertion
    #print(prompt["6"]["inputs"]["text"])
    #set the text prompt for our positive CLIPTextEncode
    prompt[mask_node_id]["inputs"]["image"] = mask_path

    #set the text prompt for our positive CLIPTextEncode
    prompt[image_node_id]["inputs"]["image"] = image_path

    #set the text prompt for our positive CLIPTextEncode
    prompt[original_image_node_id]["inputs"]["image"] = original_image_path

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt)
    ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
    #(images)
    return images

#print (sys.argv[0])
#images = run_pass(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5])
#save_images(images, sys.argv[4] or "./")


