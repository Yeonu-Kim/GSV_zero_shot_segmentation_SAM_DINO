import torch
import cv2
import os
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
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

SOURCE_IMAGE_PATH = f"{HOME}/data/satellite_ship_2.png"
image = cv2.imread(SOURCE_IMAGE_PATH)

sam_predictor.set_image(image)
masks, _, _ = sam_predictor.predict(box=None, multimask_output=False)

def masks_to_boxes(masks):
    boxes = []
    for mask in masks:
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)

xyxy_boxes = masks_to_boxes(masks)

grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

CLASSES = ['ship']
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

class_names = [f"all {cls}s" for cls in CLASSES]

detections = grounding_dino_model.predict_with_boxes(
    image=image,
    boxes=xyxy_boxes,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _, _
    in detections
]

annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
sv.plot_image(annotated_frame, (16, 16))

detections.mask = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    xyxy=detections.xyxy
)

mask_annotator = sv.MaskAnnotator()
annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

sv.plot_image(annotated_image, (16, 16))

coco_dataset = {"images":[], "annotations":[], "categories":[]}

image_info = {
    'id': 0,
    'file_name': os.path.basename(SOURCE_IMAGE_PATH),
    'width': image.shape[1],
    'height': image.shape[0]
}
coco_dataset["images"].append(image_info)

for idx, detection in enumerate(detections):
    xyxy_np, mask, _, class_id, _, _ = detection

    x_coor, y_coor = np.where(mask == 1)
    segmentation = []
    for i in range(len(x_coor)):
        segmentation.append(int(x_coor[i]))
        segmentation.append(int(y_coor[i]))

    xyxy = [float(coor) for coor in xyxy_np]

    annotation = {
        'id': idx + 1,
        'image_id': 0,
        'category_id': int(class_id),
        'segmentation': [segmentation],
        'bbox': list(xyxy),
        'area': int(np.sum(mask)),
        'iscrowd': 0
    }

    coco_dataset['annotations'].append(annotation)

categories = [{'id': i, 'name': class_name} for i, class_name in enumerate(CLASSES, 1)]
coco_dataset['categories'] = categories

with open('annotations.json', 'w') as f:
    json.dump(coco_dataset, f)
