import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
from PIL import Image
from model.faster_rcnn import FasterRCNN
from pymongo import MongoClient
import tkinter as tk
from tkinter import filedialog, messagebox

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MongoDB Setup
client = MongoClient('localhost', 27017)
db = client['image_analysis']  # Database name
collection = db['object_detection']  # Collection name (table)

# Load the trained Faster RCNN model
def load_model(model_path):
    # Define the complete model configuration as used in training
    model_config = {
        'num_classes': 6,  # Including background
        'backbone_out_channels': 512,
        'min_im_size': 600,
        'max_im_size': 1000,
        'scales': [128, 256, 512],  # Example scales
        'aspect_ratios': [0.5, 1, 2],  # Example aspect ratios
        'rpn_bg_threshold': 0.3,
        'rpn_fg_threshold': 0.7,
        'rpn_nms_threshold': 0.7,
        'rpn_train_prenms_topk': 12000,
        'rpn_test_prenms_topk': 6000,
        'rpn_train_topk': 2000,
        'rpn_test_topk': 300,
        'rpn_batch_size': 256,
        'rpn_pos_fraction': 0.5,
        'roi_iou_threshold': 0.5,
        'roi_low_bg_iou': 0.0,
        'roi_pool_size': 7,
        'roi_nms_threshold': 0.3,
        'roi_topk_detections': 100,
        'roi_score_threshold': 0.05,
        'roi_batch_size': 128,
        'roi_pos_fraction': 0.25,
        'fc_inner_dim': 1024
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FasterRCNN(model_config, num_classes=model_config['num_classes'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# Load the model
model_path = './model/faster_rcnn_address.pth' # Path to your previously trained model (extension .pth)
faster_rcnn = load_model(model_path)

# Function to process images in all subdirectories
def process_images(root_input_folder, model):
    transform = transforms.Compose([transforms.ToTensor()])
    object_id = 1  # Start counting object IDs from 1

    for subdir, dirs, files in os.walk(root_input_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                input_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, root_input_folder)

                print(f'Processing: {input_path}')  # Print the current image being processed

                image = Image.open(input_path)
                image_tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    _, output = model(image_tensor)

                # Convert output to numpy arrays
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                high_confidence_indices = scores > 0.7  # Confidence threshold

                document = {
                    "object_id": object_id,
                    "image_path": input_path,
                    "to_do": "pending"
                }

                # If any boxes have high confidence
                if high_confidence_indices.any():
                    coordinates = []

                    for box in boxes[high_confidence_indices]:
                        x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]  # Convert to list and then to int

                        # Append coordinates as a dictionary with specific keys
                        coordinates.append({
                            "upper_left": {"x": x1, "y": y1},
                            "lower_right": {"x": x2, "y": y2}
                        })

                    # Add coordinates to the document
                    document["coordinates"] = coordinates

                else:
                    # No objects detected, set coordinates to None
                    document["coordinates"] = None

                # (Optionally) Print the document to verify
                print(document)

                # Insert the document into MongoDB
                collection.insert_one(document)
                object_id += 1  # Increment object ID for the next image

    # Show a message when processing is complete
    messagebox.showinfo("Processing Complete", "All images have been processed successfully.")

# Tkinter GUI Application
def select_folder():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask the user to select a folder
    root_input_folder = filedialog.askdirectory(title="Select Folder to Process")

    if root_input_folder:
        # Confirm the selection with the user
        proceed = messagebox.askyesno("Confirm Folder", f"Do you want to process images in the folder:\n{root_input_folder}?")
        if proceed:
            # Process the images in the selected folder
            process_images(root_input_folder, faster_rcnn)
        else:
            messagebox.showinfo("Operation Cancelled", "Folder processing has been cancelled.")
    else:
        messagebox.showwarning("No Folder Selected", "No folder was selected. Please select a folder to proceed.")

if __name__ == "__main__":
    select_folder()
