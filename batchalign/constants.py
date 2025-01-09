# CHAT punctuation specifications
ENDING_PUNCT = [".", "?", "!", "+//.", "+/.", "+...", "+\"/.", "+..?", "+\".", "+//?", "+.", "+!?", "+/?", "...", "(.)"]
MOR_PUNCT = ["‡", "„", ","]
CHAT_IGNORE = ["xxx", "yyy", "www"]

# chat marks
REPEAT_GROUP_MARKS = ["[/]", "[//]", "[///]", "[////]", "[/////]", "[/-]"]
NORMAL_GROUP_MARKS = ["[?]", "[!!]", "[!]", "[>]", "[<]"]

# ud specific things for CHAT generation
UD__GENDERS = ["Masc", "Fem", "Neut", "ComNeut"]

# audio extensions
# anything in the first set but not the second set
# should be added to the third set for conversion
# we set these seperately in order to garantee fmpeg test
MEDIA_EXTENSIONS = ["*.mp3", "*.mp4", "*.wav"]
PARSABLE_MEDIA = ["*.mp3", "*.wav"]
FORCED_CONVERSION = ["mp4"] # force conversion using ffmpeg

