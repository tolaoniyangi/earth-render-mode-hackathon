import torch
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
#checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
#checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
#model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
#model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
#model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def sam2_predict(image, points, labels):
    print(f"Running sam2.1 with {points} and labels {labels}")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(point_coords=points, point_labels=labels)
        return masks