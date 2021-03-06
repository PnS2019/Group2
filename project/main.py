#!/usr/bin/python3

# import the necessary packages
print("Importing Packages", flush=True, end="")
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from responsive_voice import ResponsiveVoice
from tensorflow.keras.models import Model, load_model
import string
import numpy as np
import json
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from pySBB import get_stationboard
from pydub import AudioSegment
from pydub.generators import Sine
from pydub.playback import play
import time

print(" - Done")


def xCoo(arr):
    return arr[0][0]


def yCoo(arr):
    return arr[0][1]


def letters(picture):
    upper_bound_rectangle_area = 3000
    lower_bound_rectangle_area = 100

    potential_letters = []
    # color detection
    lower_blue = np.array([100, 40, 0])
    upper_blue = np.array([135, 255, 255])

    # converting colorspace:
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    # filtered version of Picture:
    # hsv = cv2.GaussianBlur(hsv,(3,3),2)

    filtered = cv2.inRange(hsv, lower_blue, upper_blue)
    filtered = cv2.bitwise_not(filtered)

    # contour detection:

    _, ctrs, hier = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    picture = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

    # creting an array of rectangles with areas
    rects_and_area = [[cv2.boundingRect(ctr), cv2.contourArea(ctr)] for ctr in ctrs]

    # sorting the areas and rectangles from left to right
    rects_and_area.sort(key=xCoo)

    for rectangle in rects_and_area:

        if(rectangle[1] < upper_bound_rectangle_area and rectangle[1] > lower_bound_rectangle_area):

            out_cropped = picture[rectangle[0][1]:rectangle[0][1] + rectangle[0][3],
                                  rectangle[0][0]:rectangle[0][0] + rectangle[0][2]]
            out_cropped = cv2.resize(out_cropped, (32, 32))

            potential_letters.append(out_cropped)
    return potential_letters


print("Loading machine learning model", flush=True, end="")
model = load_model('models/letclass_valacc0.921.hdf5')
print(" - Done")

# model.compile()


def get_text(letters):
    global model
    out = ""

    for letter in letters:
        pred = np.argmax(model.predict(letter.reshape(1, 32, 32, 3)), axis=1).astype(np.int)
        out += string.ascii_lowercase[pred[0]]
    return out


print("Loading tram stations", flush=True, end="")
with open("data/stations.json") as f:
    stations = json.loads(f.read())

station_list = stations.keys()
print(" - Done")


def get_station_name(station_raw):
    if station_raw != "":
        station = process.extractOne(station_raw, station_list, scorer=fuzz.ratio)
        accuracy = station[1]
        if accuracy > 70:
            # print("fuzzy: ",station_raw, station)
            return stations[station[0]]
    return None


def say_connections(station_name_full, lang=ResponsiveVoice.ENGLISH_GB):
    speaker = ResponsiveVoice(rate=.5, vol=1, gender=ResponsiveVoice.MALE, lang=lang)

    entries = get_stationboard(station_name_full)[:5]
    text = "Connections for {}. :\n".format(station_name_full)
    for entry in entries:
        if entry.category == "T":
            category = "Tram"
        else:
            category = entry.category

        seconds = int((entry.stop.departureTimestamp - time.time()))
        minutes = 1 + seconds // 60
        hours = minutes // 60
        minutes -= hours * 60

        if hours != 0:
            departure = "{} hours and {} minutes".format(hours, minutes)
        else:
            departure = "{} minutes".format(minutes)

        text += "{} Number {} to {} departs in {}.\n".format(category, entry.number, entry.to, departure)

    print(text)
    speaker.say(text)


print("Initializing Pi Camera", flush=True, end="")
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 2
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)
print("- Done")

successive_matches = 0
previous_station = None

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # Display the resulting frame
    cv2.imshow('frame', image)

    l = letters(image)
    name = get_station_name(get_text(l))

    if name == previous_station and name is not None:
        successive_matches += 1
        play(Sine(440 * successive_matches).to_audio_segment(duration=50))
    else:
        successive_matches = 0
        previous_station = name

    if successive_matches >= 2:
        say_connections(name)

    # the loop breaks at pressing `q`
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
