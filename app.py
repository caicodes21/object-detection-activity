from object_detection_helpers import *
from PIL import Image

IMAGE_FOLDER_PATH = "./assets/images/"

model, weights = initialize_model()

st.title("What is an Object Detection Model?")

st.header("Learning Objectives", divider="gray")

st.markdown('''
By the end of this activity, you will be able to:
- Define object detection model
- Create your own object detection models
''')

st.header("What is a Model?")

st.markdown("What do you see in the image below?")

st.image(IMAGE_FOLDER_PATH + "sample_objects.jpg")
st.caption("An image of everyday objects")

st.markdown('''
You likely had no trouble identifying the potted plant, apple, and bottle. 
Your eyes immediately sensed the objects and sent the visual information to your brain. Your brain 
analyzed the information, extracted the key details, and recalled the names of the objects.

How can a computer do the same thing? This is where machine learning models come in. 

A **machine learning model** is a program that is trained to spot patterns and trends in data and arrive at some 
conclusion about the data - this is basically what your brain does.

There are many types of models out there in the world, such as language models and segmentation models. 
In this activity, we will focus on **object detection models**.
''')

st.header("Try a Model Out")

st.markdown('''
Let's try running a model on the image from above. 

The model we will use is the SSDLite320 MobileNetV3 Large model from PyTorch, an open-source project used by researchers to create 
machine learning models. This model uses a feature extractor called MobileNetV3 Large, which is like a specific blueprint of 
instructions for detecting key details in an image. Without this blueprint, the model would have a hard time finding 
information that helps it spot and identify objects.

Click the button below to see what the model predicts are in the image from above.
''')

if st.button(
    label="Send to model",
    key="send sample image",
    help="Click to predict the objects in the image"
):
    st.image(detect_objects(model, weights, decode_image(IMAGE_FOLDER_PATH + "sample_objects.jpg"), 4))
    st.caption("Hooray! The model correctly found the potted plant, vase, apple, and bottle.")

st.header("Upload an Image")

st.markdown('''
Are you curious to see what the SSDLite320 MobileNetV3 Large model from PyTorch can also recognize? Try taking an image of a common object, 
like a cat, dog, computer, chair, table, or TV.

**Tip:** This app rotates the image if it is uploaded from a phone. If you are using a phone, upload a photo that was taken when the phone was 
held horziontally (not upright).
''')

uploaded_image = st.file_uploader(
    label="Upload an image of an everyday object",
    key="file uploader",
    help="Click to upload an image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_image:
    st.image(Image.open(uploaded_image).resize((256, 256)))

if st.button(
    label="Send to model",
    key="send uploaded image",
    help="Click to predict objects in your uploaded image"
) and uploaded_image:
    resized_image = Image.open(uploaded_image).resize((256, 256)).convert("RGB")
    plotted_uploaded_image = detect_objects(model, weights, T.PILToTensor()(resized_image), 1)
    st.image(plotted_uploaded_image)
    st.markdown("Did the model correctly identify the object? If not, try another image.")

st.header("How is an Oject Detection Model Made?", divider="gray")

st.markdown('''
The process of creating an object detection model is complex. Here is a simplified breakdown:

1. Gather many images for training the model (could be thousands to millions).
2. Format the images to meet certain requirements, such as specific height or width.
3. Send the formatted images to a computer that runs a training algorithm. A **training algorithm** is a sequence 
of steps for finding patterns and trends that are present in objects of the same type (ex: apples are red, 
oranges are round, cars have wheels). Over time, the computer starts to associate these patterns with 
specific objects. It practices making some predictions on the training images, and it learns from the wrong predictions.
4. Once the training algorithm is done running, the computer generates a model that can be used.
''')

st.header("Let's Make Your Own Model")

st.markdown('''
Traditionally, making an object detection model is time-consuming and costly. However, the barrier to making custom models is 
now much smaller. [Teachable Machine](https://teachablemachine.withgoogle.com) is a project that makes it 
simple for people to make models for free.

Watch this video from their website to learn more:
'''
)

st.video("https://www.youtube.com/watch?v=T2qQGqZxkD0")

st.header("Make Your Own Model")

st.markdown('''
As you can see from the video, it's relatively easy to make your own object detection model on Teachable Machine. 
Let's make one now!
''')

st.header("Step 1: Gather Some Images")

st.markdown('''
Head over to the [image training page](https://teachablemachine.withgoogle.com/train/image) of the Teachable Machine website. 

**Task:** Upload at least 10 images of two different objects, which are also known as classes. Make sure you label the names 
of the classes. Try to make every image slightly different so that the model is trained to recognize the object under different conditions, 
like different angles or lighting conditions.
''')

st.image(IMAGE_FOLDER_PATH + "step_1_image.png")
st.caption("Example of uploading images of a teddy bear (Smarty Bear :bear:) and a frog plushie (Hugging Frog :frog:)")

st.header("Step 2: Start Training")

st.markdown('''
You can start training once the images are uploaded. If you're a curious person, you might have clicked on the 
Advanced settings. The settings "epoch", "batch size", and "learning rate" are known as hyperparameters. Researchers 
typically adjust these settings manually to find the best combination that gets the most accurate model. For the sake 
of simplicity, we can use the default values.
''')

st.image(IMAGE_FOLDER_PATH + "step_2_image.png")
st.caption('''The Advanced settings can be adjusted to try to get a more accurate model, but default values are 
good enough for this activity.''')

st.markdown('''
Once training starts, your browser will run a package called Tensorflow, another open-source project 
that is commonly used to make models. Your browser uses Tensorflow to run a training algorithm. Essentially, the 
browser is training a pre-trained model to become really good at recognizing the objects that you uploaded. Think of it 
like training a chef with average cooking skills to become exceptional at specifc skills, such peeling potatoes or cutting 
a chicken. For more details, check out Teachable Machine's [FAQ](https://teachablemachine.withgoogle.com/faq).

**Task:** Click "Train Model" to start the training algorithm.
''')


st.header("Step 3: Test Your Model")

st.markdown('''
Once the training is done, you will need to send images to the model to test it. 

There are two options:

1. Webcam: Give access to your device's camera so that the model will search for the objects in the camera's live feed.
2. Manual Upload: Upload an image for testing.

Regardless of which option you chose, try to make the object look a little different from the training images. Why 
do you think it's important to make the test images not be the same as the training images? :thinking:

**Task:** Test the model using either option 1 or 2.
''')

st.image(IMAGE_FOLDER_PATH + "step_3_image.png")
st.caption("The model is tested against a never-before-seen image of Smarty Bear.")

st.markdown('''
Looks like the trained model had no problem spotting Smarty Bear :sunglasses:

You have the option to export the model or save the project. If you want to export the model or save the project, 
check out Teachable Machine's [FAQ](https://teachablemachine.withgoogle.com/faq) for more details.
'''
)



st.header("Wrap-Up", divider="gray")

st.markdown('''
Through this activity, you have learned what an object detection model is and how to make a 
small model of your own. There are many fascinating models out there. Keep learning and exploring 
the field of machine learning!
''')