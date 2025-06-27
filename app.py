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
# Assuming your refactored code is in this file
from pass_websocket import run_pass, upload_image
from segment_anything import segment_image, overlay_mask_on_image
import cv2

def flatten_masks(masks):
    """Recursively flatten all masks to a list of 2D numpy arrays."""
    flat = []
    if isinstance(masks, (list, tuple)):
        for m in masks:
            flat.extend(flatten_masks(m))
    elif hasattr(masks, "cpu") and hasattr(masks, "numpy"):
        arr = masks.cpu().numpy()
        while arr.ndim > 2:
            arr = arr[0]
        flat.append(arr)
    elif isinstance(masks, np.ndarray):
        arr = masks
        while arr.ndim > 2:
            arr = arr[0]
        flat.append(arr)
    return flat

def draw_mask_polygons_on_image(image, mask_np, color=(246, 250, 6), alpha=0.5):
    """
    Draws mask polygons on the image.
    - image: PIL.Image (RGB)
    - mask_np: 2D numpy array (mask)
    - color: tuple, RGB color for the polygon
    - alpha: float, transparency
    Returns: PIL.Image with polygons overlaid
    """
    # Convert mask to uint8
    mask_uint8 = (mask_np * 255).astype('uint8')
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert PIL image to OpenCV format
    img_cv = np.array(image).copy()
    overlay = img_cv.copy()
    # Draw filled polygons
    cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)
    # Convert back to PIL
    return Image.fromarray(img_cv)

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

if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "original_dims" not in st.session_state:
    st.session_state.original_dims = None
if "canvas_key_counter" not in st.session_state:
    st.session_state.canvas_key_counter = 0
if "sam_polygons" not in st.session_state:
    st.session_state.sam_polygons = None


# --- Callback function to handle file upload ---
def handle_upload():
    if st.session_state.file_uploader is not None:
        uploaded_file = st.session_state.file_uploader
        st.session_state.active_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_dims = st.session_state.active_image.size
        # When a new image is uploaded, clear the old polygons
        st.session_state.sam_polygons = None
        st.session_state.canvas_key_counter += 1
        st.session_state.original_image = st.session_state.active_image.copy()
    else:
        st.session_state.original_image = None
        st.session_state.active_image = None
        st.session_state.original_dims = None

# --- Streamlit UI ---
st.title("🌍🎨 Earth Canvas")
with st.container(border=True):
    st.markdown("Transform your generated designs into photorealistic renders in three simple steps!")

try:
    with open("workflow_api.json", "r") as f: WORKFLOW_TEMPLATE = json.load(f)
except: st.error("`workflow_api.json` not found! Please create it."); st.stop()


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
            # (No SAM segmentation or mask display in the sidebar)
            st.image(
                st.session_state.active_image,
                caption="Current image",
                use_container_width=True,
            )

    st.markdown("---")
    st.info("This front-end connects to a ComfyUI backend for rendering.")


# --- Main Image Editor Area ---
if st.session_state.active_image:
    col_header_1, col_header_2 = st.columns([2, 1])
    with col_header_1:
        st.subheader("2. Highlight the area to render on your canvas 🖌️")
        st.info("Draw on the canvas & right-click to complete drawing.")

        # Display canvas size (keep aspect ratio, but limit width)
        orig_w, orig_h = st.session_state.original_image.size
        if orig_w > MAX_CANVAS_WIDTH:
            disp_w, disp_h = MAX_CANVAS_WIDTH, int(MAX_CANVAS_WIDTH * (orig_h / orig_w))
        else:
            disp_w, disp_h = orig_w, orig_h

        # SAM Points workflow
        st.markdown("Click to add points for SAM segmentation.")
        point_canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_color="#FF0000",
            stroke_width=10,
            background_image=st.session_state.original_image,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="point",
            key=f"canvas_{st.session_state.canvas_key_counter}_sam",
        )

        user_points = []
        if point_canvas_result.json_data:
            for obj in point_canvas_result.json_data["objects"]:
                if obj["type"] == "circle":
                    x_disp = obj["left"] + obj["radius"]
                    y_disp = obj["top"]  + obj["radius"]
                    # Map display coords -> original image coords
                    x_orig = int(round(x_disp * (orig_w / disp_w)))
                    y_orig = int(round(y_disp * (orig_h / disp_h)))
                    user_points.append([x_orig, y_orig])

        # Button to run SAM segmentation
        if user_points:
            if st.button("🎯 Run SAM Segmentation", type="primary"):
                with st.spinner("Running SAM segmentation..."):
                    try:
                        masks, _ = segment_image(st.session_state.original_image, user_points)
                        flat_masks = flatten_masks(masks)

                        # Build polygons from masks
                        fabric_objects = []
                        for mask in flat_masks:
                            mask_u8 = (mask * 255).astype("uint8")
                            cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in cnts:
                                if cv2.contourArea(cnt) < 20:
                                    continue
                                cnt = cnt.squeeze()
                                xs, ys = cnt[:, 0], cnt[:, 1]
                                xs_disp = xs * disp_w / orig_w
                                ys_disp = ys * disp_h / orig_h

                                left, top = int(xs_disp.min()), int(ys_disp.min())
                                points = [{"x": int(xd - left), "y": int(yd - top)} for xd, yd in zip(xs_disp, ys_disp)]

                                fabric_objects.append({
                                    "type": "polygon",
                                    "version": "5.2.4",
                                    "originX": "left",
                                    "originY": "top",
                                    "left":   left,
                                    "top":    top,
                                    "width":  int(xs_disp.max() - xs_disp.min()),
                                    "height": int(ys_disp.max() - ys_disp.min()),
                                    "fill":   "rgba(246,250,6,0.4)",
                                    "stroke": "rgba(246,250,6,1.0)",
                                    "strokeWidth": 2,
                                    "points": points,
                                })

                        st.session_state.sam_polygons = {"objects": fabric_objects, "background": ""}

                    except Exception as e:
                        st.warning(f"SAM segmentation failed: {e}")
        else:
            st.info("Click on the image to add points, then run segmentation.")

        # Show editable canvas with SAM polygons (if any)
        st.markdown("### Polygons canvas – draw OR edit")
        poly_mode = st.radio(
            label="Mode:",
            options=("Draw polygons", "Move / resize"),
            horizontal=True,
            key="polygon_edit_mode",
        )

        drawing_mode_choice = "polygon" if poly_mode == "Draw polygons" else "transform"

        editable = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_color="#F6FA06",
            stroke_width=2,
            drawing_mode=drawing_mode_choice,
            height=disp_h,
            width=disp_w,
            background_image=st.session_state.original_image,
            initial_drawing=st.session_state.sam_polygons or {"objects": [], "background": ""},
            key=f"canvas_{st.session_state.canvas_key_counter}_edit",
        )

        # Store mask data for rendering
        if editable.image_data is not None:
            st.session_state.sam_mask_data = editable.image_data.copy()

    with col_header_2:
        st.subheader("3. Describe your vision ✨")

    original_w, original_h = st.session_state.original_dims
    if original_w > MAX_CANVAS_WIDTH: canvas_w, canvas_h = MAX_CANVAS_WIDTH, int(MAX_CANVAS_WIDTH * (original_h / original_w))
    else: canvas_w, canvas_h = original_w, original_h

    editor_col, controls_col = st.columns([2, 1])

    with editor_col:
        # Inform the user that polygons can be edited above.
        st.info("Use the polygons canvas above to add or edit shapes.")

    with controls_col:
        with st.container(border=True):
            prompt_text = st.text_area("Describe the desired change:", placeholder = "Aerial view of office buildings in a (neoclassical architectural style), cinematic lighting, 4k, ultra-detailed.", height=125)
            
            c1, c2 = st.columns(2)
            render_button = c1.button("Render Design", use_container_width=True, type="primary")
            if c2.button("Clear All Polygons", use_container_width=True):
                st.session_state.sam_polygons = None
                st.session_state.canvas_key_counter += 1
                st.rerun()

        if render_button:
            # Retrieve alpha channel from the SAM-editable canvas
            sam_alpha = None
            if "sam_mask_data" in st.session_state and st.session_state.sam_mask_data is not None:
                sam_alpha = st.session_state.sam_mask_data[:, :, 3]

            # Check if we have a mask to render
            has_mask = sam_alpha is not None and np.sum(sam_alpha) > 0

            if not has_mask:
                st.warning("Add points and run SAM segmentation to create a mask for rendering.")
            else:
                with st.spinner("Processing your request... This may take a moment."):
                    combined_alpha = sam_alpha

                    mask_pil = Image.fromarray(combined_alpha)
                    original_mask = mask_pil.resize(
                        st.session_state.original_dims, Image.LANCZOS
                    )

                    original_image_bytes = io.BytesIO(); st.session_state.original_image.save(original_image_bytes, format="PNG"); original_image_bytes.seek(0)
                    
                    source_image_bytes = io.BytesIO(); st.session_state.active_image.save(source_image_bytes, format="PNG"); source_image_bytes.seek(0)
                    mask_bytes = io.BytesIO(); mask_image.save(mask_bytes, format="PNG"); mask_bytes.seek(0)

                    original_path = upload_image(original_image_bytes, f"original_{CLIENT_ID}.png")
                    source_path = upload_image(source_image_bytes, f"source_{CLIENT_ID}.png")
                    mask_path = upload_image(mask_bytes, f"mask_{CLIENT_ID}.png", image_type="mask")

                    images = run_pass(prompt_text, os.path.abspath(source_path), os.path.abspath(mask_path), os.path.abspath(original_path), os.path.abspath("BuildingEditFast.json"))
                   
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
                        
                        # Clear the polygons after successful render
                        st.session_state.sam_polygons = None
                        
                        st.session_state.canvas_key_counter += 1
                        st.rerun()
                    else:
                        st.error("Render process failed to return an image.")

else:
    st.info("Upload an image using the sidebar to begin the creative process.")

