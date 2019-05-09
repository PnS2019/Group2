#!/usr/bin/python3
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

query = ' 4iblspde r%^tz'
choices = ['Albisriederplatz', 'Milchbuck', 'Hirschwiesenstrasse']
# Get a list of matches ordered by score, default limit to 5
print(query)
print(process.extract(query, choices))
# [('Barack H Obama', 95), ('Barack H. Obama', 95), ('B. Obama', 85)]

# If we want only the top one
print(process.extractOne(query, choices))
