# Object Detection Activity

Streamlit app for learning object detection

https://learn-object-detection-fun-2026.streamlit.app

## Overview

This Streamlit app is designed to provide a gentle introduction to object detection models to a non-technical audience. Users can follow the prompts in the app to learn the definiton of an object detection model and expand their learning by making their own model on [Teachable Machine](https://teachablemachine.withgoogle.com). The goal is to get people excited about learning not just object detection, but also the topic of machine learning.

## Tools Used

The app's frontend is built using Streamlit's UI components. The app lets users run the SSDLite320 MobileNetV3 Large model from PyTorch. The [example code](https://docs.pytorch.org/vision/stable/models.html#object-detection) from the PyTorch website is adapted for use in this app. Claude Code is also used to generate some of the code in this project. Teachable Machine's platform is used to teach users how to create custom object detection models.

This project uses the font Roboto downloaded from Google Fonts, licensed under the SIL Open Font License. It is used to label the names of objects detected by the SSDLite320 model.
