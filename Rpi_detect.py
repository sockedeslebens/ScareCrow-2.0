# Based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/README.md
import re
import cv2
import play_sound
import time
from tflite_runtime.interpreter import Interpreter
import numpy as np
from multiprocessing import Process
from picamera2 import Picamera2


CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 768
THRESHOLD = 0.7   # set threshold for object detection

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels



def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims((image-255)/255, axis=0)
  


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor



def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)

  results = []
  for i in range(len(scores)):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def main():
    
    labels = load_labels()
    interpreter = Interpreter('detect.tflite')
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    cv2.startWindowThread()
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (CAMERA_WIDTH, CAMERA_HEIGHT)}))
    picam2.start()
    time.sleep(2)

    last_rec_time = time.time()
    # set time interval for object detection
    time_delta = 5

    while True:

      frame = picam2.capture_array()
      curr_time = time.time()

      if (curr_time- last_rec_time) >= time_delta:

        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640,640))
        res = detect_objects(interpreter, img,THRESHOLD)
        
        if 'p' in globals():
          p.join()

        # get coordinates of bounding boxes
        for result in res:
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            
            cv2.rectangle(frame,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.putText(frame,labels[int(result['class_id'])],(xmin, min(ymax, CAMERA_HEIGHT-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 
        
        last_rec_time = curr_time

      else:
        res = []
      
      cv2.imshow('Pi Feed', frame)

      # trigger sound when object is detected
      if res:
          p = Process(target=play_sound.play)
          p.start()
          play_sound.play()

      if cv2.waitKey(10) & 0xFF==ord('q'):
          cv2.destroyAllWindows()
          break

if __name__ == "__main__":
    main()