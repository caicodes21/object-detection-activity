import torch
import torchvision
import streamlit as st
from torchvision.io.image import decode_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T

FONT_TTF_PATH = "./assets/fonts/Roboto-VariableFont_wdth,wght.ttf"

## fetch the model weights and initialize the model
## cache the weights to avoid re-fetching
@st.cache_resource
def initialize_model():
    weights = torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.9)
    model.eval()
    return [model, weights]

## helper function for predicting objects in an image
@torch.no_grad()
def detect_objects(model, weights, image, prediction_count):
    preprocess = weights.transforms()
    batch = [preprocess(image)]
    prediction = model(batch)[0]
    scores = prediction["scores"]
    top_prediction_indices = scores.argsort(descending=True)[:prediction_count]
    labels = [weights.meta["categories"][i] for i in prediction["labels"][top_prediction_indices]]
    box = draw_bounding_boxes(
        image, 
        boxes=prediction["boxes"][top_prediction_indices],
        labels=labels,
        colors="black",
        width=8,
        font=FONT_TTF_PATH,
        font_size=20
    )
    plotted_image = to_pil_image(box.detach())
    return plotted_image

