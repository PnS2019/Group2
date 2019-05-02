#!/usr/bin/python3

with open("stations.csv") as f:
    stations_raw = f.read().split("\n")

stations = []

for station_raw in stations_raw:
    station = station_raw.split(", ")[-1].lower()
