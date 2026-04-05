# Object Detection Activity

## Overview

This Streamlit app is designed to provide a gentle introduction to object detection models for a non-technical audience. Users can follow the prompts in the app to learn the definiton of an object detection model and expand their learning by making their own model on [Teachable Machine](https://teachablemachine.withgoogle.com). The goal is to get people excited about learning not just object detection, but also the topic of machine learning.

## Tools Used

The app's frontend is built using Streamlit's UI components. The app lets users run a Faster R-CNN RestNet50-FPN v2 model from PyTorch. The [example code](https://docs.pytorch.org/vision/stable/models.html#object-detection) from the PyTorch website is adapted for use in this app. The app contains explanations of how to use Teachable Machine to create custom object detection models.

This project uses the font Roboto downloaded from Google Fonts, licensed under the SIL Open Font License. It is used to label the names of objects detected by the Faster R-CNN model.
