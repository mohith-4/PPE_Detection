

import os
import cv2
import gradio as gr
from ultralytics import YOLO

# Load class names (assuming you have a class_names.txt file)
with open("class_names.txt", "r") as f:
    class_names = [name.strip() for name in f.readlines()]

# Function to load the YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to process a single image
def process_image(person_model, ppe_model, img, conf_threshold=0.25):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    original_img = img.copy()

    # Person detection
    person_results = person_model(img)

    for person in person_results[0].boxes:
        if person.conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, person.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]

            # PPE detection
            ppe_results = ppe_model(cropped_img)
            for ppe in ppe_results[0].boxes:
                if ppe.conf >= conf_threshold:
                    px1, py1, px2, py2 = map(int, ppe.xyxy[0])

                    # Coordinates for drawing on the original image
                    full_x1 = x1 + px1
                    full_y1 = y1 + py1
                    full_x2 = x1 + px2
                    full_y2 = y1 + py2

                    confidence = float(ppe.conf)
                    class_id = int(ppe.cls)
                    class_name = class_names[class_id]

                    label = f"{class_name} {confidence:.2f}"

                    cv2.rectangle(original_img, (full_x1, full_y1), (full_x2, full_y2), (0, 255, 0), 2)
                    cv2.putText(original_img, label, (full_x1, full_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert back to RGB for display
    return cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

# Function to run inference
def run_inference(person_model_path, ppe_model_path, img):
    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)

    if person_model is None or ppe_model is None:
        return "Error: Model loading failed."

    return process_image(person_model, ppe_model, img)

# Gradio Interface Setup
title = "YOLO Person and PPE Detection"
description = "A demo to detect persons and their personal protective equipment (PPE) using YOLO models."
examples = [["example_images/example1.jpg"], ["example_images/example2.jpg"]]

person_model_path = "path_to_person_model.pt"  # specify your model paths
ppe_model_path = "path_to_ppe_model.pt"

gr.Interface(
    fn=lambda img: run_inference(person_model_path, ppe_model_path, img),
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title=title,
    description=description,
    examples=examples
).launch()
