def get_character_set(transcripts):
    """
    Gets distincts characters for all dataset
    :param transcripts:
    :return:
    """
    characters_set = set()
    for t in transcripts:
        for c in t:
            if c not in characters_set:
                characters_set.add(c)
    return characters_set


def get_longest_sample_size(transcripts):
    """
    Return the maximum sample length for our dataset
    :param transcripts: List of String
    :return: int
    """
    return max([len(transcript) for transcript in transcripts])


def convert_to_int(character_set):
    """
    Returns a dict containing the int that corresponds to the char to encode input
    :param character_set: set
    :return: dict
    """
    char_to_int = dict()
    for i, char in enumerate(character_set):
        char_to_int[char] = i
    return char_to_int


def convert_to_char(character_set):
    """
        Returns a dict containing the char that corresponds to an int to decode target
        :param character_set: set
        :return: dict
        """
    int_to_char = dict()
    for i, char in enumerate(character_set):
        int_to_char[i] = char

    return int_to_char


def convert_int_to_char(char_to_int):
    int_to_char = {}
    for key, value in char_to_int.items():
        int_to_char[value] = key
    return int_to_char


def get_distinct_words(transcripts):
    distinct_words = []
    for transcript in transcripts:
        for word in transcript.split():
            if word not in distinct_words:
                distinct_words.append(word)

    return distinct_words


def convert_words_to_int(distinct_words):
    word_to_int = dict()
    int_to_word = dict()
    for i, word in enumerate(distinct_words):
        word_to_int[word] = i
        int_to_word[i] = word
    return word_to_int, int_to_word


def decode_transcript(encoded_transcript, character_set):
    """
    Takes and encoded transcript and the character_set of our transcripts corpus and outputs the transcript as String type
    :param encoded_transcript: numpy 2d array
    :param character_set: Dict
    :return: String
    """
    transcript = ""
    conversion_table = convert_to_char(character_set)
    encoded_transcript = encoded_transcript.astype(int)
    for character in encoded_transcript:
        for index, value in enumerate(list(character)):
            if character[index] == 1:
                transcript += conversion_table[index]

    return transcript
