from responsive_voice import ResponsiveVoice
from pydub import AudioSegment
from pydub.playback import play

english_speaker = ResponsiveVoice(rate=.5, vol=1, gender=ResponsiveVoice.FEMALE, lang=ResponsiveVoice.ENGLISH_GB)
german_speaker = ResponsiveVoice(rate=.5, vol=1, gender=ResponsiveVoice.FEMALE, lang=ResponsiveVoice.GERMAN)

file1 = english_speaker.get_mp3("Tram Number 10 to ")
file2 = german_speaker.get_mp3("Zürich, Flughafen")
file3 = english_speaker.get_mp3("departs at 13:45")

cutaway_ms = 150
segment1 = AudioSegment.from_mp3(file1)[:-cutaway_ms]
segment2 = AudioSegment.from_mp3(file2)[cutaway_ms:-cutaway_ms]
segment3 = AudioSegment.from_mp3(file3)[cutaway_ms:]

total = segment1 + segment2 + segment3
play(total)
