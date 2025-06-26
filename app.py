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
from pass_websocket import run_pass, upload_image

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Earth Canvas", page_icon="🌍")

# --- CSS Injection for Theme AND Google Sans Font ---
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css?family=Google+Sans:400,500,700');

        p {
            font-family: 'Google Sans', sans-serif;
        }

        .block-container { padding-top: 2rem; }
        [data-testid="stSidebar"] { background-color: #3C4043; }
        .stButton>button {
            font-family: 'Google Sans', sans-serif;
            color: #202124;
            background-color: #8AB4F8;
            border: 1px solid #8AB4F8;
        }
        h1 {
            font-family: 'Google Sans', sans-serif;
            color: #8AB4F8 !important;
        }
        h2, h3{
            font-family: 'Google Sans', sans-serif;
            color: #AECBFA !important;
        }
        h1, p { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App State and Constants ---
#COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
CLIENT_ID = str(uuid.uuid4())
MAX_CANVAS_WIDTH = 800

if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "original_dims" not in st.session_state:
    st.session_state.original_dims = None


def get_image(bytes):
    try:
        return Image.open(io.BytesIO(bytes))
    except:
        st.error("Error getting image.")
        return None


# --- NEW: Callback function to handle the file upload ---
# This function runs ONLY when the file uploader's value changes.
def handle_upload():
    if st.session_state.file_uploader is not None:
        uploaded_file = st.session_state.file_uploader
        st.session_state.active_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_dims = st.session_state.active_image.size
    else:
        # This handles the case where the user clicks 'x' to clear the file
        st.session_state.active_image = None
        st.session_state.original_dims = None


# --- Streamlit UI ---
st.title("🌍🎨 Earth Canvas")
with st.container(border=True):
    st.markdown(
        "Transform your generated designs into photorealistic renders in three simple steps!"
    )

try:
    with open("workflow_api.json", "r") as f:
        WORKFLOW_TEMPLATE = json.load(f)
except:
    st.error("`workflow_api.json` not found! Please create it.")
    st.stop()


# --- Sidebar for Upload and Control ---
with st.sidebar:
    st.header("1. Upload Image 🖼️")
    with st.container(border=True):
        # --- CHANGE: Using the on_change callback pattern ---
        st.file_uploader(
            "Upload new or replacement image.",
            type=["png", "jpg", "jpeg"],
            key="file_uploader",  # A static key is better for callbacks
            on_change=handle_upload,  # This is the key change!
        )

        # --- CHANGE: Display logic is now separate from upload logic ---
        # It reads from session_state, which is set by the callback.
        # This prevents the duplication issue.
        if st.session_state.active_image:
            st.image(
                st.session_state.active_image,
                caption="Current image",
                use_container_width=True,
            )

    st.markdown("---")
    st.info("This front-end connects to a ComfyUI backend to perform the rendering.")


# --- Main Image Editor Area ---
if st.session_state.active_image:
    col_header_1, col_header_2 = st.columns([2, 1])
    with col_header_1:
        st.subheader("2. Highlight the area to render on your canvas 🖌️")
    with col_header_2:
        st.subheader("3. Describe your vision ✨")

    original_w, original_h = st.session_state.original_dims
    if original_w > MAX_CANVAS_WIDTH:
        canvas_w, canvas_h = MAX_CANVAS_WIDTH, int(
            MAX_CANVAS_WIDTH * (original_h / original_w)
        )
    else:
        canvas_w, canvas_h = original_w, original_h

    editor_col, controls_col = st.columns([2, 1])

    with editor_col:
        canvas_result = st_canvas(
            fill_color="rgba(138, 180, 248, 0.4)",
            stroke_width=20,
            background_color="#3C4043",
            background_image=st.session_state.active_image,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="freedraw",
            key="canvas",
        )

    with controls_col:
        with st.container(border=True):
            prompt_text = st.text_area(
                "****",
                "Aerial view of office buildings in a (neoclassical architectural style), cinematic lighting, 4k, ultra-detailed.",
                height=125,
            )
            render_button = st.button(
                "Render Design", use_container_width=True, type="primary"
            )

        if render_button:
            if (
                canvas_result.image_data is None
                or np.sum(canvas_result.image_data[:, :, 3]) == 0
            ):
                st.warning(
                    "Please draw a mask on the image to indicate the area to render."
                )
            else:
                with st.spinner("Processing your request... This may take a moment."):
                    mask_data = canvas_result.image_data[:, :, 3]
                    mask_pil = Image.fromarray(mask_data)
                    original_mask = mask_pil.resize(
                        st.session_state.original_dims, Image.LANCZOS
                    )

                    source_image_bytes = io.BytesIO()
                    st.session_state.active_image.save(source_image_bytes, format="PNG")
                    source_image_bytes.seek(0)
                    mask_bytes = io.BytesIO()
                    original_mask.save(mask_bytes, format="PNG")
                    mask_bytes.seek(0)

                    source_filename = f"source_{CLIENT_ID}.png"
                    mask_filename = f"mask_{CLIENT_ID}.png"
                    source_path = upload_image(source_image_bytes, source_filename)
                    mask_path = upload_image(mask_bytes, mask_filename, image_type="mask")

                    images = run_pass(prompt_text, os.path.abspath(source_path), os.path.abspath(mask_path), os.path.abspath("BuildingEditv2.json"))
                   
                    for node_id in images:
                        print(node_id)
                        for image_data in images[node_id]:
                            print(dir(image_data))
                            #count = count + 1
                            #filename = f"tempoutput{count}.png"
                            image = Image.open(io.BytesIO(image_data))

                            final_image = image
                            if final_image != None:
                                st.success("Enhancement complete!")
                                st.session_state.active_image = final_image
                                st.session_state.original_dims = final_image.size
                                st.rerun()
                                break
                            else:
                                print("Image did not parse")

else:
    st.info("Please upload an image using the sidebar to begin the creative process.")
