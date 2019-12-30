# Sign Language Recognition

This project is used to detect and predict hand gestures with the help of OpenCV and TensorFlow

## Getting Started

### Requirements

1. Python 3.x
2. OpenCV 4.1.0
3. TensorFlow 1.13
4. Numpy

## Installing

Download / Clone the project into your local machine. I recommend creating a virtual environment using conda as it worked perfectly for me and the execute the following command.

`pip3 install -r requirements.txt`

## Setup

### 1\. Creating hand histogram

If you want to generate your own histogram for the purpose of collecting your own images follow these steps

`python generate_hand_histogram.py`

After executing this command in the terminal, a window will open on the screen. If it doesn't there might be a problem with the webcam. The default camera is set to webcam, if you are using a secondary camera with this project you need to change the following code in `generate_hand_histogram.py` file

```
cap = cv2.VideoCapture(0) to cap = cv2.VideoCapture(1)
```

After the window open properly place your hand over the green boxes and make sure you cover all of them to generate a good histogram. On window follow the instructions.

Press 'c' key on the keyboard to capture the histogram. If you think you captured a decent histogram, press key 's' on keyboard to save the histogram. If the histogram was not captured properly you can again place the hand on the green box and press 'c' on keyboard to capture it again.

### 2\. Generate Images for training

Once the histogram is generated it will we saved in the project directory as hand_histogram.pickle

To collect the hand images for training execute the following command:

`python collect_training_images.py` [label_name] [num_samples]

Both the command line argument are necessary

- label_name : The name of your label
- num_samples: Total number of samples you want to collect

After executing this command a window will open press 's' key on keyboard to start saving the images or 'q' key on keyboard to quit

After collecting all the images the window will be closed automatically

Collect 200 images each of at least 5-6 labels to train your model properly
