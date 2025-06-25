import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import requests
import json
import websocket
import uuid
import io
import os
import numpy as np

# --- Configuration ---
st.set_page_config(layout="wide")

# Set this to the URL of your ComfyUI server.
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
CLIENT_ID = str(uuid.uuid4())
MAX_CANVAS_WIDTH = 800 # Max width for the editor canvas in pixels

# --- Session State Initialization ---
# This is the core of our iterative editing. We store the current image here.
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "original_dims" not in st.session_state:
    st.session_state.original_dims = None
# A key for the file uploader to allow resetting it.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- ComfyUI API Interaction Functions (Same as before) ---
def queue_prompt(prompt_workflow):
    try:
        req = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt_workflow, "client_id": CLIENT_ID})
        req.raise_for_status()
        return req.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error queuing prompt: {e}")
        return None

def upload_image(image_bytes, filename, image_type="input"):
    try:
        files = {'image': (filename, image_bytes, 'image/png')}
        data = {'overwrite': 'true', 'type': image_type}
        resp = requests.post(f"{COMFYUI_URL}/upload/image", files=files, data=data)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading image: {e}")
        return None

def get_image(filename, subfolder, folder_type):
    try:
        url = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting image: {e}")
        return None

def get_history(prompt_id):
    try:
        with requests.get(f"{COMFYUI_URL}/history/{prompt_id}") as response:
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting history: {e}")
        return None

def get_final_image_from_ws():
    # Use ws:// or wss:// depending on the main URL scheme
    ws_scheme = "ws" if "http://" in COMFYUI_URL else "wss"
    ws_url = f"{ws_scheme}://{COMFYUI_URL.split('//')[1]}/ws?clientId={CLIENT_ID}"
    
    ws = websocket.WebSocket()
    try:
        ws.connect(ws_url)
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing' and message['data']['node'] is None:
                    return get_history(message['data']['prompt_id'])
    except (websocket.WebSocketException, ConnectionRefusedError) as e:
        st.error(f"Websocket connection error: {e}. Is the server running and accessible at {COMFYUI_URL}?")
        return None
    finally:
        if ws.connected:
            ws.close()

# --- Streamlit UI ---

st.title("🎨 Iterative Image Enhancement")
st.markdown("Upload an image to start. Then, mask an area, describe your changes, and click Enhance. The result will become the new image, ready for more edits.")

# Load the workflow template
try:
    with open("workflow_api.json", "r") as f:
        WORKFLOW_TEMPLATE = json.load(f)
except FileNotFoundError:
    st.error("`workflow_api.json` not found! Please create it.")
    st.stop()

# --- Sidebar for Upload and Control ---
with st.sidebar:
    st.header("1. Start or Reset")
    uploaded_file = st.file_uploader(
        "Upload a new image",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file:
        # When a new image is uploaded, it becomes the active image.
        st.session_state.active_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_dims = st.session_state.active_image.size
        st.success("Image uploaded! You can now start editing in the main area.")

    if st.button("Start Over with a New Image"):
        # Clear the state and increment the key to reset the file uploader widget
        st.session_state.active_image = None
        st.session_state.original_dims = None
        st.session_state.uploader_key += 1
        st.rerun() # Rerun the script to reflect the changes

    st.markdown("---")
    st.info("This front-end connects to a ComfyUI backend to perform the rendering.")


# --- Main Image Editor Area ---
if st.session_state.active_image:
    st.header("2. Edit Your Image")
    
    # --- NEW: Image Scaling Logic ---
    # We scale the image to fit the canvas, and later scale the mask back up.
    original_w, original_h = st.session_state.original_dims
    
    # Calculate scaled dimensions to fit within MAX_CANVAS_WIDTH
    if original_w > MAX_CANVAS_WIDTH:
        aspect_ratio = original_h / original_w
        canvas_w = MAX_CANVAS_WIDTH
        canvas_h = int(canvas_w * aspect_ratio)
    else:
        canvas_w, canvas_h = original_w, original_h
    
    # The canvas now uses the scaled dimensions
    editor_col, controls_col = st.columns([2, 1])

    with editor_col:
        st.write("Draw a mask over the area you want to change.")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 255, 0.3)",
            stroke_width=20,
            background_image=st.session_state.active_image, # Always use the active image
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="freedraw",
            key="canvas",
        )

    with controls_col:
        st.subheader("3. Guide the AI")
        prompt_text = st.text_area(
            "Describe the desired change:",
            "A beautiful golden chalice, 4k, hyperrealistic, masterpiece",
            height=100,
        )

        if st.button("✨ Enhance Image", use_container_width=True, type="primary"):
            if canvas_result.image_data is None or np.sum(canvas_result.image_data[:, :, 3]) == 0:
                st.warning("Please draw a mask on the image to indicate the area of interest.")
            else:
                with st.spinner("Processing your request..."):
                    # 1. Get the mask from the canvas and scale it up to original dimensions
                    mask_data = canvas_result.image_data[:, :, 3] # Alpha channel is the mask
                    mask_pil = Image.fromarray(mask_data)
                    # --- NEW: Resizing the mask ---
                    original_mask = mask_pil.resize(st.session_state.original_dims, Image.LANCZOS)

                    # 2. Convert active image and final mask to bytes for upload
                    source_image_bytes = io.BytesIO()
                    st.session_state.active_image.save(source_image_bytes, format='PNG')
                    source_image_bytes.seek(0)
                    
                    mask_bytes = io.BytesIO()
                    original_mask.save(mask_bytes, format='PNG')
                    mask_bytes.seek(0)

                    # 3. Upload images to ComfyUI
                    st.write("Uploading images to backend...")
                    source_filename = f"source_{CLIENT_ID}.png"
                    mask_filename = f"mask_{CLIENT_ID}.png"
                    upload_image(source_image_bytes, source_filename)
                    upload_image(mask_bytes, mask_filename, image_type="mask")

                    # 4. Prepare and queue the prompt
                    prompt_workflow = WORKFLOW_TEMPLATE.copy()
                    prompt_workflow["6"]["inputs"]["text"] = prompt_text
                    prompt_workflow["10"]["inputs"]["image"] = source_filename
                    prompt_workflow["14"]["inputs"]["image"] = mask_filename
                    prompt_workflow["3"]["inputs"]["seed"] = int(uuid.uuid4().int & (1<<32)-1)

                    st.write("Queueing prompt and waiting for render...")
                    prompt_data = queue_prompt(prompt_workflow)
                    
                    if prompt_data:
                        prompt_id = prompt_data['prompt_id']
                        output_history = get_final_image_from_ws()
                        
                        if output_history:
                            history = output_history[prompt_id]
                            output_data = history['outputs']['8']['images'][0]
                            final_image = get_image(
                                output_data['filename'], 
                                output_data['subfolder'], 
                                output_data['type']
                            )
                            
                            st.success("Enhancement complete!")
                            
                            # --- NEW: This is the core of the iterative loop ---
                            # The new output becomes the active image for the next round.
                            st.session_state.active_image = final_image
                            st.session_state.original_dims = final_image.size # Update dims too
                            st.rerun() # Rerun the script to update the canvas with the new image
                        else:
                            st.error("Failed to get a result from the backend. Check server logs.")
                    
else:
    st.info("Please upload an image using the sidebar to begin the editing process.")