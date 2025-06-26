import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np

# Prefer CUDA if available, otherwise fall back to Apple-Silicon/Metal (MPS) when present, and finally CPU.
if torch.cuda.is_available():
    _device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(_device)
_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

def segment_image(image, input_points):
    """
    Segments objects in the given image using SAM.

    Args:
        image (PIL.Image): The input image.
        input_points (list): List of list of 2D points [[x, y], ...] for segmentation prompts.

    Returns:
        masks: Segmentation masks (list of numpy arrays)
        scores: IOU scores (torch.Tensor)
    """
    # Ensure the points are floats (necessary for torch.float32 casting)
    input_points = [[float(x), float(y)] for x, y in input_points]

    # Prepare inputs
    inputs = _processor(image, input_points=[input_points], return_tensors="pt")

    # Cast any float64 tensors to float32 to guarantee MPS compatibility
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v) and v.dtype == torch.float64:
            inputs[k] = v.to(dtype=torch.float32)

    # Move tensors to the selected device
    inputs = {k: (v.to(_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    masks = _processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    scores = outputs.iou_scores.cpu()
    return masks, scores

def overlay_mask_on_image(image, mask, color=(30, 144, 255), alpha=0.6):
    """
    Overlay a binary mask on a PIL image.
    - image: PIL.Image (RGB)
    - mask: numpy array (H, W), values 0 or 1
    - color: tuple, RGB color for the mask overlay
    - alpha: float, transparency of the mask
    Returns: PIL.Image (RGBA)
    """
    # Ensure mask is uint8
    mask = (mask * 255).astype('uint8')
    mask_img = Image.fromarray(mask, mode='L').resize(image.size)
    color_img = Image.new('RGBA', image.size, color + (0,))
    # Put the mask as the alpha channel
    color_img.putalpha(mask_img.point(lambda p: int(p * alpha)))
    # Convert original image to RGBA
    base = image.convert('RGBA')
    # Composite
    return Image.alpha_composite(base, color_img)

def get_mask_image(mask):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img

if __name__ == "__main__":
    # Example usage
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    input_points = [[450, 600]]  # Example: 2D location of a window in the image
    masks, scores = segment_image(raw_image, input_points)
    print("Masks:", masks)
    print("Scores:", scores)

"""
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`
"""