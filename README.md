# PythonAIDetect

[About](#about-data-anonymizer)

[Installation](#installation)

[Usage](#usage)

[MongoDB database](#mongodb-database)

[Troubleshooting during development](#troubleshooting-during-development)

[Licence](#licence)


# About PythonAIDetect

"PythonAIDetect" is an locally run application where a user, through simple 'tkinter' interface, can choose folder with jpg images to process. The application uses previously trained Faster-RCNN (Faster Region Based Convolutional Neural Network) type neural network, which was traind for looking hand written mail addresses, and sends metdata about every image to MongoDB database. 

It sends to MongoDB database named 'image_analysis', and its collection 'object_detection' that kind of [metadata](#mongodb-database).

The script for training Faster-RCNN model of neural network is taken from [explaininai](https://github.com/explainingai-code/FasterRCNN-PyTorch?tab=readme-ov-file) github repository.

# Installation

To run this application on Linux you need to have installed:
- Python 3.10.12, or higher
- MongoDB - I used Docker for it
- Download this repository

Structure of the application should look like this:
```bash
PythonAIDetect
|-----config
|     |-----voc.yaml
|
|-----model
|     |-----__pycache__
|     |-----faster_rcnn.py
|     |-----__init__.py
|     |-----your_previously_trained_model (you should have your own trained model here; file with .pth extension)
|
|-----PythonDetect.py
|-----.gitignore
|-----README.md
```


# Usage 

- In Linux terminal go to main folder of 'PythonAIDetect' and type down **python PythonDetect.py**. This command will start the application.

![pic_p](https://github.com/user-attachments/assets/e5dd26ff-ae5e-4b79-88f5-fd4ac76c28d1)

The user has to enter a folder with images he wants to process. During the process all necessary informations are displayed in a terminal.

# MongoDB database

Example of one document of metadata on MongoDB database sent by the application:
```bash
  {
    "_id": "6715569eeecdefd4d8ab3df7",
    "object_id": 9,
    "image_path": "/path/to/folder/with/images/PL_11_FOK_399_2_15_0024.jpg",
    "to_do": "pending",
    "coordinates": [
      {
        "upper_left": {
          "x": 1305,
          "y": 232
        },
        "lower_right": {
          "x": 2284,
          "y": 775
        }
      }
    ]
  }
```

The application creates: 
- 'image path' the place where the image is found on a disk.
- 'coordinates' where the sensitive data can be. 
- Default value of 'to_do' is 'pending'. Later it is changed after modifications done by the user. It helps to avoid browsing images, which were modified earlier.


# Troubleshooting during development
While loading the model I had to add the same configuration as it had been used in a script for training the neural network: 
```
    model_config = {
        'num_classes': 6,
        'backbone_out_channels': 512,
        'min_im_size': 600,
        'max_im_size': 1000,
        'scales': [128, 256, 512],
        'aspect_ratios': [0.5, 1, 2],
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
```
Initially I did not have all necessary keys that RCNN model expected. And that caused 'keyError'.

# Licence
This project is open source.
