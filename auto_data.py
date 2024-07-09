import torch
import cv2
import os
import sys
import supervision as sv
import matplotlib.pyplot as plt
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import math
from pycocotools import coco, cocoeval, mask as cocomask
import json

HOME = os.getcwd()

GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))

def add_all_suffix(class_names):
    result = []
    for class_name in class_names:
        result.append("all " + class_name + "s")
    return result

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = "vit_h"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

SOURCE_IMAGE_PATH = f"{HOME}/data/__FwscCqAFl8v3uNUXf4ow.jpeg"
CLASSES = ['pole']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

class_names = add_all_suffix(CLASSES)

# load image
image = cv2.imread(SOURCE_IMAGE_PATH)

for class_name in class_names:
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=[class_name],
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{class_name} {confidence:0.2f}" 
        for _, _, confidence, _, _, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    sv.plot_image(annotated_frame, (16, 16))

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # Create a combined mask
    combined_mask = np.zeros_like(image[:, :, 0])  # Create a blank image with the same height and width as the original image, but single channel

    # Overlay each mask onto the blank image
    for mask in detections.mask:
        combined_mask = np.maximum(combined_mask, mask)

    # Save the combined mask as an image file
    combined_mask_path = f"{HOME}/combined_mask.png"
    cv2.imwrite(combined_mask_path, combined_mask * 255)  # Multiply by 255 to convert the binary mask to an 8-bit image

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{class_name} {confidence:0.2f}"
        for _, _, confidence, _, _, _
        in detections
    ]

    #if you dont update SV you will face problems here .
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    sv.plot_image(annotated_image, (16, 16))

    grid_size_dimension = math.ceil(math.sqrt(len(detections.mask)))

    titles = [
        CLASSES[class_id]
        for class_id
        in detections.class_id
    ]

    # Calculate the number of rows and columns for the grid
    grid_size_dimension = math.ceil(math.sqrt(len(detections.mask)))

    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(grid_size_dimension, grid_size_dimension, figsize=(16, 16))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate over each image and plot it on the corresponding subplot
    for idx, image in enumerate(detections.mask):
        ax = axes[idx]
        ax.imshow(image, cmap='gray')
        ax.set_title(titles[idx])
        ax.axis('off')  # Hide the axis

    # Hide any remaining empty subplots if the number of images is less than the grid size
    for idx in range(len(detections.mask), len(axes)):
        axes[idx].axis('off')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    print(f"Combined mask saved at {combined_mask_path}")
