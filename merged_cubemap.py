import torch
import cv2
import os
import sys
import supervision as sv
import matplotlib.pyplot as plt
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import pandas as pd
import math
from pycocotools import coco, cocoeval, mask as cocomask
from util.converter import panorama_to_cubemap, merge_faces
import json
from tqdm import tqdm
import warnings
import random
from PIL import Image

warnings.filterwarnings(action='ignore')

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

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray, dominant: bool=True) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True,
            dominant=dominant
        )
        if dominant:
            index = np.argmax(scores)
        else:
            sums = [np.sum(mask) for mask in masks]
            index = np.argmin(sums)
        # index = np.argmax(scores)
        
        result_masks.append(masks[index])
    return np.array(result_masks)


grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = "vit_h"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

SOURCE_IMAGE_PATH = "./streetview_data"
# SOURCE_IMAGE_PATH = r"./data"
CLASSES = ['tree', 'shrub', 'building', 'sky', 'road', 'sidewalk', 'vehicle', 'telegraph pole', 'cable', 'trash', 'bench']
TINY_CLASSES = ['cable']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

class_names = add_all_suffix(CLASSES)
tiny_objects = add_all_suffix(TINY_CLASSES)

# Save the result in a dataframe
df_columns = ['name'] + CLASSES
df = pd.DataFrame(columns = df_columns)

# load image
# images = [os.path.join(SOURCE_IMAGE_PATH, f) for f in tqdm(os.listdir(SOURCE_IMAGE_PATH)) if os.path.isfile(os.path.join(SOURCE_IMAGE_PATH, f))]
# images.sort()
# images = random.sample(images, 900)
images = ["_5CZ_jWgfVrMCSLGghjCpw.jpeg", "_9IbrnZ5Ebnk-i71yozsDA.jpeg", "_CMohzZzxbwRFBrQnNSVtg.jpeg"]
images = [os.path.join(SOURCE_IMAGE_PATH, f) for f in images]

for turn, image_path in enumerate(tqdm(images)):
    raw_image = cv2.imread(image_path)
    filename, extension = os.path.basename(image_path).rsplit('.', 1)
    if extension != 'jpeg':
        continue

    total_pixel = np.zeros(len(CLASSES))

    cubemaps = panorama_to_cubemap(raw_image)
    image = merge_faces(cubemaps)[...,::-1]

    original_image_path = f"{HOME}/output/cubemap/original/{filename}.jpeg"
    image_data = Image.fromarray(image)
    image_data.save(original_image_path,'JPEG')

    for class_idx, class_name in enumerate(class_names):
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

        # sv.plot_image(annotated_frame, (16, 16))

        # convert detections to masks
        if class_name in tiny_objects:
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy,
                dominant = False
            )
        else: 
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

        total_pixel[class_idx] += np.sum(combined_mask)

        # Save the combined mask as an image file
        # if turn % 100 == 0:
        #     combined_mask_path = f"{HOME}/output/{class_name}/{filename}.png"
        #     cv2.imwrite(combined_mask_path, combined_mask * 255)  # Multiply by 255 to convert the binary mask to an 8-bit image

        combined_mask_path = f"{HOME}/output/cubemap/{CLASSES[class_idx]}/{filename}.jpeg"
        mask_img = Image.fromarray(combined_mask*255)
        mask_img.save(combined_mask_path,'JPEG')

        # Show image polts for debugging
        if len(detections.mask) == 0:
            continue

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

        # sv.plot_image(annotated_image, (16, 16))

        grid_size_dimension = math.ceil(math.sqrt(len(detections.mask)))

        titles = [
            CLASSES[class_id]
            for class_id
            in detections.class_id
        ]

        # Calculate the number of rows and columns for the grid
        grid_size_dimension = math.ceil(math.sqrt(len(detections.mask)))

        # # Display the combined_mask image
        # plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        # plt.imshow(combined_mask, cmap='gray')  # Use 'gray' colormap for grayscale images
        # plt.title('Combined Mask')
        # plt.axis('off')  # Hide axis ticks and labels
        # plt.tight_layout()
        # plt.show()

        # print(f"Combined mask saved at {combined_mask_path}")

    total_pixel /= (image.shape[0] * image.shape[1])
    df.loc[len(df)] = [filename] + list(total_pixel)

output_file_path = "./output/output_cubemap_merged.csv"
df.to_csv(output_file_path, index=False)
print(df)
