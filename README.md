# MilkTea Brand Recognition System
A simple image classification project based on **PyTorch + ResNet18 + Flask**.

## Features
- Train a milk tea brand classifier with ResNet18
- Predict uploaded images
- Web demo with Flask
- Show prediction result and confidence score

## Brands
- coco
- heytea
- mixue
- naixue

## Project Structure
text
milktea_project/
│── train.py
│── predict.py
│── app.py
│── best_model.pth
│── dataset/
│── templates/
│── uploads/
│── README.md


## Installation
pip install torch torchvision flask pillow

## Train Model
python train.py

## Predict Image
Put a test image in the folder and run:
bash 
python predict.py

## Run Web Demo
bash 
python app.py

Open browser:
http://127.0.0.1:5000

## Notes

* Dataset and model weights are ignored in `.gitignore`
* This project is for learning purposes

