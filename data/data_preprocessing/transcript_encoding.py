import gc

import numpy as np
from bitarray import bitarray

from etc import settings
from utils import convert_to_int, get_longest_word_length
from ..dataset_generation import generate_pickle_file
from utils import get_empty_binary_vector, convert_word_to_binary


def _get_transcriptions(audioInput_data):
    """
    Returns transcripts of the dataset
    :return: List of Strings
    """
    transcripts = []
    for audio_sample in audioInput_data:
        transcripts.append(audio_sample.audio_transcript)
    return transcripts


def one_hot_encode_transcript(transcript, char_to_int, num_distinct_chars):
    """
    One hot encodes a transcript to input_transcript and target_transcript which is the input_transcript at t+1
    :param transcript: String
    :param char_to_int: Dict
    :param num_distinct_chars: Int
    :return: Numpy 2dArray, Numpy 2dArray
    """
    input_transcript = np.zeros((len(transcript),
                                 num_distinct_chars),
                                dtype='float32')

    target_transcript = np.zeros((len(transcript),
                                  num_distinct_chars),
                                 dtype='float32')
    for index, character in enumerate(transcript):
        input_transcript[index, char_to_int[character]] = 1
        if index > 0:
            input_transcript[index - 1, char_to_int[character]] = 1

    return input_transcript, target_transcript


def _generate_character_input_target_data(transcripts, char_to_int, num_distinct_chars):
    """
    Generates two 3D arrays for the decoder input data and target data.
    Fills the 3D arrays with each sample of our dataset
    Returns OneHotEncoded Decoder Input data
    Returns OneHotEncoded Target data
    :param transcripts: List of Strings
    :param char_to_int: Dict
    :param num_distinct_chars: int
    :return: 3D numpy Array, 3D numpy Array
    """
    encoded_decoder_inputs = []
    encoded_decoder_targets = []

    for transcript in transcripts:
        encoded_input, encoded_target = one_hot_encode_transcript(transcript=transcript,
                                                                  char_to_int=char_to_int,
                                                                  num_distinct_chars=num_distinct_chars)
        encoded_decoder_inputs.append(encoded_input)
        encoded_decoder_targets.append(encoded_target)

    decoder_input_data = np.array(encoded_decoder_inputs)
    decoder_target_data = np.array(encoded_decoder_targets)

    return decoder_input_data, decoder_target_data


def _generate_variable_size_character_input_target_data(transcripts, char_to_int):
    """
        Generates two 3D arrays for the decoder input data and target data.
        Fills the 3D arrays for each sample of our dataset
        Return OneHotEncoded Decoder Input data
        Return OneHotEncoded Target data
        :param transcripts: List of Strings
        :param char_to_int: Dict
        :return: 3D numpy Array, 3D numpy Array
        """

    # Init numpy array
    num_transcripts = len(transcripts)
    decoder_input_data = np.array([None] * num_transcripts)
    decoder_target_data = np.array([None] * num_transcripts)

    for i, transcript in enumerate(transcripts):
        # Encode each transcript
        encoded_transcript_input = []
        encoded_transcript_target = []

        for index, character in enumerate(transcript):
            # Encode each character
            encoded_character = [0] * len(char_to_int)
            encoded_character[char_to_int[character]] = 1

            encoded_transcript_input.append(encoded_character)
            encoded_transcript_target.append([])

            if index > 0:
                encoded_transcript_target[index - 1] = encoded_character

        del encoded_transcript_input[-1]
        decoder_input_data[i] = encoded_transcript_input

        encoded_transcript_target.pop()
        decoder_target_data[i] = encoded_transcript_target

    return decoder_input_data, decoder_target_data


def generate_variable_size_character_input_target_data(transcripts, num_partition, char_to_int, partitions=32,
                                                       test=False):
    """
        Generates two 3D arrays for the decoder input data and target data.
        Fills the 3D arrays for each sample of our dataset
        Return OneHotEncoded Decoder Input data
        Return OneHotEncoded Target data
        :param transcripts: List of Strings
        :param char_to_int: Dict
        :return: 3D numpy Array, 3D numpy Array
        """

    # Dividing transcripts into subsets
    transcript_sets = []
    limits = []
    for i in range(1, partitions + 1):
        limits.append(int(len(transcripts) * i / partitions))

    transcript_sets.append(transcripts[0: limits[0]])
    for i in range(1, partitions):
        transcript_sets.append(transcripts[limits[i - 1]:limits[i]])

    # Delete original dataset
    transcripts = []
    gc.collect()

    for num_dataset, transcript_set in enumerate(transcript_sets):
        # Init numpy array
        num_transcripts = len(transcript_set)
        decoder_input_data = np.array([None] * num_transcripts)
        decoder_target_data = np.array([None] * num_transcripts)

        for i, transcript in enumerate(transcript_set):
            # Encode each transcript
            encoded_transcript_input = []
            encoded_transcript_target = []

            for index, character in enumerate(transcript):
                # Encode each character
                encoded_character = [0] * len(char_to_int)
                encoded_character[char_to_int[character]] = 1

                encoded_transcript_input.append(encoded_character)
                encoded_transcript_target.append([])

                if index > 0:
                    encoded_transcript_target[index - 1] = encoded_character

            del encoded_transcript_input[-1]
            decoder_input_data[i] = encoded_transcript_input
            encoded_transcript_target.pop()
            decoder_target_data[i] = encoded_transcript_target
        if not test:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH + "dataset" + str(
                num_partition) + "/encoded_transcripts" + str(num_dataset) + ".pkl"
        else:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH + "dataset" + str(
                num_partition) + "/encoded_transcripts" + str(num_dataset) + ".pkl"

        generate_pickle_file((decoder_input_data, decoder_target_data), file_path=path)

    # return decoder_input_data, decoder_target_data


def generate_variable_word_based_encoding_final(transcripts, num_partition, char_to_int, partitions=8, test=False):
    longest_word_length = get_longest_word_length(settings.WORD_SET)
    transcript_sets = []
    limits = []
    for i in range(1, partitions + 1):
        limits.append(int(len(transcripts) * i / partitions))

    transcript_sets.append(transcripts[0: limits[0]])
    for i in range(1, partitions):
        transcript_sets.append(transcripts[limits[i - 1]:limits[i]])

    # Delete original dataset
    transcripts = []
    gc.collect()
    word_sizes = dict()

    for num_dataset, transcript_set in enumerate(transcript_sets):
        # Init numpy array
        num_transcripts = len(transcript_set)
        decoder_input_data = np.array([None] * num_transcripts)
        decoder_target_data = np.array([None] * num_transcripts)
        character_set_length = len(settings.CHARACTER_SET) + 1
        for i, transcript in enumerate(transcript_set):
            # Encode each transcript
            encoded_transcript_input = []
            encoded_transcript_target = []
            list_words = transcript.split()

            for index, word in enumerate(list_words):
                if len(word) not in word_sizes:
                    word_sizes[len(word)] = 0
                word_sizes[len(word)] += 1

                encoded_word = [0] * settings.WORD_TARGET_LENGTH
                # encoded_word = []
                encoded_word_target = []
                for j in range(0, longest_word_length):
                    # encoded_word.append([0] * len(settings.CHARACTER_SET))
                    encoded_word_target.append([0] * character_set_length)

                character_index = 0
                for character_index, character in enumerate(word):
                    # Encoding words for decoder inputs
                    position = char_to_int[character] + character_set_length * character_index
                    encoded_word[position] = 1
                    # Encoding word for decoder targets
                    encoded_word_target[character_index][char_to_int[character]] = 1

                if character_index < longest_word_length - 1:

                    for k in range(character_index + 1, longest_word_length):
                        position = character_set_length * k + character_set_length - 1
                        encoded_word[position] = 1

                        encoded_word_target[k][-1] = 1
                encoded_transcript_input.append(encoded_word)
                encoded_transcript_target.append([])
                if index > 0:
                    encoded_transcript_target[index - 1] = encoded_word_target

            del encoded_transcript_input[-1]
            decoder_input_data[i] = encoded_transcript_input
            encoded_transcript_target.pop()
            decoder_target_data[i] = encoded_transcript_target

        if not test:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH + "dataset" + str(
                num_partition) + "/encoded_transcripts" + str(num_dataset) + ".pkl"
        else:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH + "dataset" + str(
                num_partition) + "/encoded_transcripts" + str(num_dataset) + ".pkl"

        generate_pickle_file((decoder_input_data, decoder_target_data), file_path=path)


def generate_variable_word_based_encoding(transcripts, num_partition, char_to_int, partitions=8, test=False):
    print(char_to_int)
    print(settings.WORD_TARGET_LENGTH)
    transcript_sets = []
    limits = []
    for i in range(1, partitions + 1):
        limits.append(int(len(transcripts) * i / partitions))

    transcript_sets.append(transcripts[0: limits[0]])
    for i in range(1, partitions):
        transcript_sets.append(transcripts[limits[i - 1]:limits[i]])

    # Delete original dataset
    transcripts = []
    gc.collect()

    for num_dataset, transcript_set in enumerate(transcript_sets):
        # Init numpy array
        num_transcripts = len(transcript_set)
        decoder_input_data = np.array([None] * num_transcripts)
        decoder_target_data = np.array([None] * num_transcripts)
        for i, transcript in enumerate(transcript_set):
            # Encode each transcript
            encoded_transcript_input = []
            encoded_transcript_target = []
            list_words = transcript.split()
            for index, word in enumerate(list_words):
                encoded_word = [0] * settings.WORD_TARGET_LENGTH  # this is 882
                for character_index, character in enumerate(word):
                    position = char_to_int[character] + len(settings.CHARACTER_SET) * character_index  # working fine
                    encoded_word[position] = 1

                encoded_transcript_input.append(encoded_word)
                encoded_transcript_target.append([])
                if index > 0:
                    encoded_transcript_target[index - 1] = encoded_word

            del encoded_transcript_input[-1]
            decoder_input_data[i] = encoded_transcript_input
            encoded_transcript_target.pop()
            decoder_target_data[i] = encoded_transcript_target

        if not test:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH + "encoded_transcripts" + str(num_dataset) + ".pkl"
        else:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH + "encoded_transcripts" + str(num_dataset) + ".pkl"

        generate_pickle_file((decoder_input_data, decoder_target_data), file_path=path)


def generate_variable_word_input_target_binary(transcripts, words_to_int, partitions=8, test=False):
    transcript_sets = []
    limits = []
    for i in range(1, partitions + 1):
        limits.append(int(len(transcripts) * i / partitions))

    transcript_sets.append(transcripts[0: limits[0]])
    for i in range(1, partitions):
        transcript_sets.append(transcripts[limits[i - 1]:limits[i]])

    # Delete original dataset
    transcripts = []
    gc.collect()

    for num_dataset, transcript_set in enumerate(transcript_sets):
        # Init numpy array
        num_transcripts = len(transcript_set)
        decoder_input_data = np.array([None] * num_transcripts)
        decoder_target_data = np.array([None] * num_transcripts)

        for i, transcript in enumerate(transcript_set):
            # Encode each transcript
            encoded_transcript_input = []
            encoded_transcript_target = []
            list_words = transcript.split()
            for index, word in enumerate(list_words):
                word_index = words_to_int[word]
                output_binary_vector = get_empty_binary_vector(len(words_to_int))
                encoded_word = convert_word_to_binary(word_index, output_binary_vector)

                encoded_transcript_input.append(encoded_word)
                encoded_transcript_target.append([])
                if index > 0:
                    encoded_transcript_target[index - 1] = encoded_word

            del encoded_transcript_input[-1]
            decoder_input_data[i] = encoded_transcript_input
            encoded_transcript_target.pop()
            decoder_target_data[i] = encoded_transcript_target

        if not test:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH + "encoded_transcripts" + str(num_dataset) + ".pkl"
        else:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH + "encoded_transcripts" + str(num_dataset) + ".pkl"

        generate_pickle_file((decoder_input_data, decoder_target_data), file_path=path)


def generate_variable_word_input_target_data(transcripts, words_to_int, partitions=8, test=False):
    # Dividing transcripts into subsets
    transcript_sets = []
    limits = []
    for i in range(1, partitions + 1):
        limits.append(int(len(transcripts) * i / partitions))

    transcript_sets.append(transcripts[0: limits[0]])
    for i in range(1, partitions):
        transcript_sets.append(transcripts[limits[i - 1]:limits[i]])

    # Delete original dataset
    transcripts = []
    gc.collect()

    for num_dataset, transcript_set in enumerate(transcript_sets):
        # Init numpy array
        num_transcripts = len(transcript_set)
        decoder_input_data = np.array([None] * num_transcripts)
        decoder_target_data = np.array([None] * num_transcripts)

        for i, transcript in enumerate(transcript_set):
            # Encode each transcript
            encoded_transcript_input = []
            encoded_transcript_target = []
            list_words = transcript.split()
            for index, word in enumerate(list_words):
                # Encode each character
                encoded_word = bitarray(len(list_words))

                # encoded_word = bytearray([0] * len(words_to_int))
                encoded_word[words_to_int[word]] = 1

                encoded_transcript_input.append(encoded_word)
                encoded_transcript_target.append([])

                if index > 0:
                    encoded_transcript_target[index - 1] = encoded_word

            del encoded_transcript_input[-1]
            decoder_input_data[i] = encoded_transcript_input
            encoded_transcript_target.pop()
            decoder_target_data[i] = encoded_transcript_target

        if not test:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH + "encoded_transcripts" + str(num_dataset) + ".pkl"
        else:
            path = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH + "encoded_transcripts" + str(num_dataset) + ".pkl"
        generate_pickle_file((decoder_input_data, decoder_target_data), file_path=path)


def generate_decoder_input_target(transcripts, dataset_number, word_level=False, test=False,
                                  partitions=32):
    """
    Wrapper for the _generate_input_target_data method.
    :return: 3D numpy Array, 3D numpy Array
    """

    if word_level:
        character_set = settings.CHARACTER_SET
        char_to_int = convert_to_int(sorted(character_set))
        generate_variable_word_based_encoding_final(transcripts=transcripts,
                                                    num_partition=dataset_number,
                                                    char_to_int=char_to_int,
                                                    partitions=partitions,
                                                    test=test)

        # generate_variable_word_input_target_binary(transcripts=transcripts,
        #                                           words_to_int=word_to_int,
        #                                           partitions=partitions,
        #                                           test=test)

    else:
        # Character level recognition
        character_set = settings.CHARACTER_SET
        char_to_int = convert_to_int(sorted(character_set))
        generate_variable_size_character_input_target_data(transcripts=transcripts,
                                                           num_partition=dataset_number,
                                                           char_to_int=char_to_int,
                                                           partitions=partitions,
                                                           test=test)
