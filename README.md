# PythonAIDetect

[About](#about-data-anonymizer)

[Installation](#installation)

[Usage](#usage)

[MongoDB database](#mongodb-database)

[Troubleshooting during development](#troubleshooting-during-development)

[Licence](#licence)


# About PythonAIDetect

"PythonAIDetect" is an locally run application where a user, through simple 'tkinter' interface, can choose folder with jpg images to process. The application uses previously trained RCNN type neural network, which was traind for looking hand written mail addresses, and sends metdata about every image to MongoDB database.

It sends to MongoDB database named 'image_analysis', and its collection 'object_detection' that kind of [metadata](#mongodb-database).

The RCNN (Reginal Convolutional Neural Network) model of neural network is based on https

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

![pic1](https://github.com/user-attachments/assets/3c9e82a5-6929-4762-a6e5-eec79b5df0c0)
Initial screen with main menu.



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
- 'coordinates' where the sensitive data can be. After the user's modifications this section is upgraded.
- After the user's modifications updates 'to_do' from 'pending' to 'done'. It helps to avoid browsing images, which were modified earlier.


# Troubleshooting during development


# Licence
This project is open source.
