from responsive_voice import ResponsiveVoice
from pydub import AudioSegment
from pydub.playback import play
from pySBB import get_stationboard
import time


def say_connections_onelang(station_name_full, lang=ResponsiveVoice.ENGLISH_GB):
    speaker = ResponsiveVoice(rate=.5, vol=1, gender=ResponsiveVoice.FEMALE, lang=lang)

    entries = get_stationboard(station_name_full)[:5]
    text = "Connections for {}:\n".format(station_name_full)
    for entry in entries:
        if entry.category == "T":
            category = "Tram"
        else:
            category = entry.category

        print(entry.stop.departureTimestamp, time.time())
        seconds = int((entry.stop.departureTimestamp - time.time()))
        minutes = 1 + int(seconds / 60)
        hours = minutes // 60
        minutes -= hours * 60
        seconds -= hours * 3600 + minutes * 60

        if hours != 0:
            departure = "{} hours and {} minutes".format(hours, minutes)
        else:
            departure = "{} minutes".format(minutes)

        text += "{} Number {} to {} departs in {}.\n".format(category, entry.number, entry.to, departure)

    print(text)
    speaker.say(text)


say_connections_onelang("Zürich, Milchbuck")
