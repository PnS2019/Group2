# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from responsive_voice import ResponsiveVoice
from tensorflow.keras.models import Model, load_model
import string


def xCoo(arr):
    return arr[0][0]


def yCoo(arr):
    return arr[0][1]


def letters(picture):
    potential_letters = []
    # color detection
    lower_blue = np.array([100, 40, 0])
    upper_blue = np.array([135, 255, 255])

    # converting colorspace:
    picture = cv.resize(picture, (1920, 1080))
    hsv = cv.cvtColor(picture, cv.COLOR_BGR2HSV)
    # filtered version of Picture:
    #hsv = cv.GaussianBlur(hsv,(3,3),2)

    filtered = cv.inRange(hsv, lower_blue, upper_blue)
    filtered = cv.bitwise_not(filtered)

    # contour detection:

    _, ctrs, hier = cv.findContours(filtered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    picture = cv.cvtColor(filtered, cv.COLOR_GRAY2RGB)

    # creting an array of rectangles with areas
    rects_and_area = [[cv.boundingRect(ctr), cv.contourArea(ctr)] for ctr in ctrs]

    # sorting the areas and rectangles from left to right
    rects_and_area.sort(key=xCoo)

    for rectangle in rects_and_area:

        if(rectangle[1] < upper_bound_rectangle_area and rectangle[1] > lower_bound_rectangle_area):

            out_cropped = picture[rectangle[0][1]:rectangle[0][1] + rectangle[0][3],
                                  rectangle[0][0]:rectangle[0][0] + rectangle[0][2]]
            out_cropped = cv.resize(out_cropped, (32, 32))

            potential_letters.append(out_cropped)
    return potential_letters


def say_text(text):
    """Speaks a text over the speaker"""
    speaker = ResponsiveVoice(rate=.5, vol=1)
    speaker.say(text, gender="male", lang="en-GB")


model = load_model('models/letclass_valacc0.921.hdf5')
model.compile()


def get_text(letters):
    global model
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

    l = letters(image)
    name = get_text(l)
    print(name)

    # Display the resulting frame
    cv2.imshow('frame', image)
    # the loop breaks at pressing `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
