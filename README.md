# ScareCrow-2.0
This project was developed to automatically scare away pigeons from a balcony to prevent nesting using a Raspberry Pi Model 4.
A TensorFlow SSD-MobileNet V2 FPNLite (640x640) model for object detection was trained to detect pigeons in crowded scenes. 
For deployment on a Raspberry Pi the TensorFlow model was converted to a TensorFloe Lite flat buffer file (detect.tflite).
Upon detection of a pigeon in the camera feed sounds will be triggered to scare away pigeons.

## Usage
Clone this repository and install the required packages. 
`git clone git@github.com:sockedeslebens/ScareCrow-2.0.git`

### Requirements
The following packages are required for the scripts to work.
  + opencv-python
  + tflite-runtime
  + pygame
  + picamera2
  
To install these packages, you can simply run:
`pip install -r requirements.txt`

Finally call `python Rpi_detect.py`to run the script.



  
