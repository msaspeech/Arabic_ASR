import re

from lang_trans.arabic import buckwalter
from num2words import num2words


def arabic_to_buckwalter(arabic_sentence):
    return buckwalter.transliterate(arabic_sentence)


def buckwalter_to_arabic(buckwalter_sentence):
    return buckwalter.untransliterate(buckwalter_sentence)


def convert_numeral_to_written_number(number):
    written_number = num2words(number, lang="ar")
    pattern = "واحد ألف"
    matches = re.findall(pattern, written_number)
    if matches:
        to_replace_with = "ألف"
        written_number = re.sub(matches[0], to_replace_with, written_number)

    return arabic_to_buckwalter(written_number)


def numerical_to_written_numbers_table(inf_number=0, sup_number=10001):
    mapped_numbers = {}
    for i in range(inf_number, sup_number):
        mapped_numbers[i] = arabic_to_buckwalter(convert_numeral_to_written_number(i))
    return mapped_numbers
