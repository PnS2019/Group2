#!/usr/bin/python3
import json
from fuzzywuzzy import process

with open("data/stations.json") as f:
    stations = json.loads(f.read())

station_list = stations.keys()

results = process.extract("miick", station_list, limit=1)


print(stations[results[0][0]], results)
