import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import requests
import json
import websocket
import uuid
import io
import os
import numpy as np
# Make sure shapely is in your requirements.txt
from shapely.geometry import Polygon
from shapely.ops import unary_union
# Assuming your refactored code is in this file
from pass_websocket import run_pass, upload_image

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Earth Canvas", page_icon="🌍")

# --- CSS Injection ---
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css?family=Google+Sans:400,500,700');
        p { font-family: 'Google Sans', sans-serif; }
        .block-container { padding-top: 2rem; }
        [data-testid="stSidebar"] { background-color: #3C4043; }
        .stButton>button { font-family: 'Google Sans', sans-serif; color: #202124; background-color: #8AB4F8; border: 1px solid #8AB4F8; }
        h1 { font-family: 'Google Sans', sans-serif; color: #8AB4F8 !important; }
        h2, h3{ font-family: 'Google Sans', sans-serif; color: #AECBFA !important; }
        h1, p { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App State and Constants ---
CLIENT_ID = str(uuid.uuid4())
MAX_CANVAS_WIDTH = 800

if "active_image" not in st.session_state: st.session_state.active_image = None
if "original_dims" not in st.session_state: st.session_state.original_dims = None
# --- NEW: A persistent list to store polygons that have been rendered ---
if "rendered_polygons" not in st.session_state: st.session_state.rendered_polygons = []
if "canvas_key_counter" not in st.session_state: st.session_state.canvas_key_counter = 0

# --- Callback function to handle file upload ---
def handle_upload():
    if st.session_state.file_uploader is not None:
        uploaded_file = st.session_state.file_uploader
        st.session_state.active_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_dims = st.session_state.active_image.size
        # When a new image is uploaded, clear the old polygons
        st.session_state.rendered_polygons = []
        st.session_state.canvas_key_counter += 1
    else:
        st.session_state.active_image = None
        st.session_state.original_dims = None

# --- Streamlit UI ---
st.title("🌍🎨 Earth Canvas")
with st.container(border=True):
    st.markdown("Transform your generated designs into photorealistic renders in three simple steps!")

try:
    with open("workflow_api.json", "r") as f: WORKFLOW_TEMPLATE = json.load(f)
except: st.error("`workflow_api.json` not found! Please create it."); st.stop()

with st.sidebar:
    st.header("1. Upload Image 🖼️")
    with st.container(border=True):
        st.file_uploader("Upload new or replacement image.", type=["png", "jpg", "jpeg"], key="file_uploader", on_change=handle_upload)
        if st.session_state.active_image: st.image(st.session_state.active_image, caption="Current image", use_container_width=True)
    st.markdown("---")
    st.info("This front-end connects to a ComfyUI backend for rendering.")

if st.session_state.active_image:
    col_header_1, col_header_2 = st.columns([2, 1])
    with col_header_1: st.subheader("2. Highlight areas to render 🖌️")
    with col_header_2: st.subheader("3. Describe your vision ✨")

    original_w, original_h = st.session_state.original_dims
    if original_w > MAX_CANVAS_WIDTH: canvas_w, canvas_h = MAX_CANVAS_WIDTH, int(MAX_CANVAS_WIDTH * (original_h / original_w))
    else: canvas_w, canvas_h = original_w, original_h

    editor_col, controls_col = st.columns([2, 1])

    with editor_col:
        st.info("Click points to create polygons. Double-click to finish a shape. Previously rendered areas are outlined.")
        
        # --- NEW: Prepare the initial drawing with outlines only ---
        initial_drawing = {"objects": []}
        for obj in st.session_state.rendered_polygons:
            # Create a copy to modify its properties for display
            display_obj = obj.copy()
            display_obj["fill"] = "rgba(0,0,0,0)"  # Transparent fill
            display_obj["stroke"] = "#F6FA06"      # Visible outline
            initial_drawing["objects"].append(display_obj)
            
        canvas_result = st_canvas(
            # New polygons will be filled
            fill_color="rgba(255, 255, 255, 0.5)",
            stroke_color="#F6FA06",
            stroke_width=5,
            background_color="#000000",
            background_image=st.session_state.active_image,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="polygon",
            # Feed the "outline-only" polygons to the canvas
            initial_drawing=initial_drawing,
            key=f"canvas_{st.session_state.canvas_key_counter}",
        )

    with controls_col:
        with st.container(border=True):
            prompt_text = st.text_area("Describe the desired change:", "Aerial view of office buildings in a (neoclassical architectural style), cinematic lighting, 4k, ultra-detailed.", height=125)
            
            c1, c2 = st.columns(2)
            render_button = c1.button("Render Design", use_container_width=True, type="primary")
            if c2.button("Clear All Polygons", use_container_width=True):
                st.session_state.rendered_polygons = []
                st.session_state.canvas_key_counter += 1
                st.rerun()

        if render_button:
            if canvas_result.json_data is None or not canvas_result.json_data.get("objects"):
                st.warning("Please draw at least one new polygon to render.")
            else:
                with st.spinner("Processing your request... This may take a moment."):
                    mask_image = Image.new("L", (original_w, original_h), 0)
                    draw = ImageDraw.Draw(mask_image)
                    scale_x = original_w / canvas_w
                    scale_y = original_h / canvas_h
                    
                    shapely_polygons = []
                    for obj in canvas_result.json_data["objects"]:
                        # CHANGE THIS LINE
                        if obj['type'] == 'path': # Or whatever the actual type string is for your shapes
                            points = [(p[1] * scale_x, p[2] * scale_y) for p in obj['path'] if len(p) == 3]
                            if len(points) > 2:
                                shapely_polygons.append(Polygon(points))

                    if not shapely_polygons:
                        st.warning("No valid polygons found. Ensure shapes have at least 3 points."); st.stop()

                    merged_geometry = unary_union(shapely_polygons)

                    # Draw the final, merged shape to the mask
                    if merged_geometry.geom_type == 'Polygon': geoms = [merged_geometry]
                    elif merged_geometry.geom_type == 'MultiPolygon': geoms = list(merged_geometry.geoms)
                    else: geoms = []
                    
                    for poly in geoms:
                        draw.polygon(list(poly.exterior.coords), fill=255)
                        for interior in poly.interiors: draw.polygon(list(interior.coords), fill=0)

                    # ... (rest of the code for sending to backend is the same)
                    source_image_bytes = io.BytesIO(); st.session_state.active_image.save(source_image_bytes, format="PNG"); source_image_bytes.seek(0)
                    mask_bytes = io.BytesIO(); mask_image.save(mask_bytes, format="PNG"); mask_bytes.seek(0)

                    source_path = upload_image(source_image_bytes, f"source_{CLIENT_ID}.png")
                    mask_path = upload_image(mask_bytes, f"mask_{CLIENT_ID}.png", image_type="mask")

                    images = run_pass(prompt_text, os.path.abspath(source_path), os.path.abspath(mask_path), os.path.abspath("BuildingEditv2.json"))
                   
                    final_image = None
                    if images:
                        for node_id in images:
                            for image_data in images[node_id]:
                                final_image = Image.open(io.BytesIO(image_data)); break
                            if final_image: break
                    
                    if final_image:
                        st.success("Enhancement complete!")
                        st.session_state.active_image = final_image
                        st.session_state.original_dims = final_image.size
                        # --- KEY CHANGE: Save ALL polygons for the next pass ---
                        st.session_state.rendered_polygons = canvas_result.json_data["objects"]
                        st.session_state.canvas_key_counter += 1
                        st.rerun()
                    else:
                        st.error("Render process failed to return an image.")

else:
    st.info("Upload an image using the sidebar to begin the creative process.")