# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from responsive_voice import ResponsiveVoice
from tensorflow.keras.models import Model, load_model
import string


def say_text(text):
  """Speaks a text over the speaker"""
  speaker = ResponsiveVoice(rate=.5, vol=1)
  speaker.say(text, gender="male", lang="en-GB")


def get_text(letters):
  model = load_model('my_model.h5')
  model.compile()
  preds = np.argmax(model.predict(letters), axis=1).astype(np.int)
  out = ""
  for letter_index in preds:
    out += string.ascii_lowercase[letter_index]
  return out



# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  # grab the raw NumPy array representing the image, then initialize the timestamp
  # and occupied/unoccupied text
  image = frame.array

  # Display the resulting frame
  cv2.imshow('frame', image)
  # the loop breaks at pressing `q`
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # clear the stream in preparation for the next frame
  rawCapture.truncate(0)
