from torchvision.io.image import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

FONT_TTF_PATH = "./assets/fonts/Roboto-VariableFont_wdth,wght.ttf"

## fetch the model weights and initialize the model
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
model.eval()

## helper function for predicting objects in an image
def detect_objects(model, weights, image):

    preprocess = weights.transforms()
    batch = [preprocess(image)]
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(
        image, 
        boxes=prediction["boxes"],
        labels=labels,
        colors="blue",
        width=8,
        font=FONT_TTF_PATH,
        font_size=30
    )
    plotted_image = to_pil_image(box.detach())

    return plotted_image

