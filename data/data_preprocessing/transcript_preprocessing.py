import re

from utils import convert_numeral_to_written_number, arabic_to_buckwalter


def special_characters_table():
    special_characters = dict()
    special_characters["%"] = arabic_to_buckwalter("في المئة")
    special_characters["@"] = ""
    special_characters["#"] = ""
    special_characters[";"] = ""
    special_characters["B"] = "b"
    special_characters["C"] = ""
    special_characters["e"] = "E"
    special_characters["G"] = "g"
    special_characters["I"] = "i"
    special_characters["J"] = "j"
    special_characters["L"] = "l"
    special_characters["M"] = "m"
    special_characters["O"] = "o"
    special_characters["P"] = "p"
    special_characters["Q"] = "q"
    special_characters["R"] = "r"
    special_characters["U"] = "u"
    special_characters["V"] = "v"
    special_characters["W"] = "w"
    special_characters["X"] = "x"
    special_characters["ﻻ"] = ""
    special_characters["ﻹ"] = ""
    special_characters["ﻷ"] = ""
    special_characters["ﻵ"] = ""
    special_characters["ﺇ"] = ""
    special_characters["٠"] = convert_numeral_to_written_number(0)
    special_characters["١"] = convert_numeral_to_written_number(1)
    special_characters["٢"] = convert_numeral_to_written_number(2)
    special_characters["٣"] = convert_numeral_to_written_number(3)
    special_characters["٤"] = convert_numeral_to_written_number(4)
    special_characters["٦"] = convert_numeral_to_written_number(6)
    special_characters["٩"] = convert_numeral_to_written_number(9)

    return special_characters


def _replace_numbers(transcription):
    pattern = "\d+"
    numbers = re.findall(pattern, transcription)

    if numbers:
        print("match result is : " + str(numbers))
        for number in numbers:
            to_replace_with = convert_numeral_to_written_number(int(number))
            print("------" + number + "-----" + to_replace_with)
            transcription = re.sub(number, to_replace_with, transcription)

    return transcription


def _replace_special_characters(transcription, special_characters_table):
    for pattern, to_replace_with in special_characters_table.items():
        matches = re.findall(pattern, transcription)
        if matches:
            for match in matches:
                print("------" + match + "-----" + to_replace_with)
                transcription = re.sub(match, to_replace_with, transcription)

    return transcription


def _remove_noisy_numbers(transcription):
    numbers = []
    for i in range(0, 10):
        numbers.append(str(i))
    for c in transcription:
        if c in numbers:
            transcription = re.sub(c, "", transcription)
    return transcription


def transcript_preprocessing(transcription, special_characters_table):
    # replacing special characters

    new_transcription = _replace_special_characters(transcription, special_characters_table)

    # Replacing numbers
    new_transcription = _replace_numbers(new_transcription)

    # deleting remaining numbers
    new_transcription = _remove_noisy_numbers(new_transcription)

    if transcription == new_transcription:
        return transcription
    else:
        print("TO BE REMOVED")
        return None
