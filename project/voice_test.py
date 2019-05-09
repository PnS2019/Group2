from responsive_voice import ResponsiveVoice


def say_text(text, lang="en-GB"):
    """Speaks a text over the speaker"""
    speaker = ResponsiveVoice(rate=.5, vol=1)
    speaker.say(text, gender="male", lang=lang)


say_text("Tram Number 10 to ", lang="en-GB")
say_text("Zürich, Flughafen", lang="de-DE")
say_text("departs at 13:45", lang="en-GB")
