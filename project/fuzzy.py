#!/usr/bin/python3
import json
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from pySBB import get_stationboard
from responsive_voice import ResponsiveVoice


def say_text(text):
    """Speaks a text over the speaker"""
    speaker = ResponsiveVoice(rate=.5, vol=1)
    speaker.say(text, gender="male", lang="en-GB")


with open("data/stations.json") as f:
    stations = json.loads(f.read())

station_list = stations.keys()

station_raw = "miibuck"


def say_connections():
    results = process.extractOne(station_raw, station_list, scorer=fuzz.ratio)

    station_name = results[0]

    station_name_full = stations[station_name]
    entries = get_stationboard(station_name_full)[:5]
    text = "Connections for {}:\n".format(station_name_full)
    for entry in entries:
        if entry.category == "T":
            category = "Tram"
        else:
            category = entry.category
        text += "{} Number {} to {}, departs at {}.\n".format(category, entry.number, entry.to, entry.stop.departure)

    print(text)
    say_text(text)
