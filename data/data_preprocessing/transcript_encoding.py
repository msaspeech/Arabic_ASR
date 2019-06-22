import gc

import numpy as np
from bitarray import bitarray

from etc import settings
from utils import convert_to_int, get_longest_word_length
from ..dataset_generation import generate_pickle_file
from utils import get_empty_binary_vector, convert_word_to_binary


def generate_character_level_input_target_data(transcripts, num_partition, char_to_int, partitions,
                                               test=False):
    """
        Generates two 3D arrays for the decoder input data and target data.
        Fills the 3D arrays for each sample of our dataset
        Return OneHotEncoded Decoder Input data
        Return OneHotEncoded Target data
        :param num_partition: integer
        :param partitions: List
        :param test: Boolean
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


def generate_word_level_input_target_data(transcripts, num_partition, char_to_int, partitions, test=False):
    """
           Generates two 3D arrays for the decoder input data and target data.
           Fills the 3D arrays for each sample of our dataset
           :param num_partition: Integer
           :param partitions: List
           :param test: Boolean
           :param transcripts: List of Strings
           :param char_to_int: Dict
           :return: 3D numpy Array, 3D numpy Array
           """
    longest_word_length = get_longest_word_length(settings.WORD_SET)
    transcript_sets = []
    limits = []
    for i in range(1, partitions + 1):
        limits.append(int(len(transcripts) * i / partitions))

    transcript_sets.append(transcripts[0: limits[0]])
    for i in range(1, partitions):
        transcript_sets.append(transcripts[limits[i - 1]:limits[i]])

    # Delete original dataset
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
        # Word level encoding
        character_set = settings.CHARACTER_SET
        char_to_int = convert_to_int(sorted(character_set))
        generate_word_level_input_target_data(transcripts=transcripts,
                                              num_partition=dataset_number,
                                              char_to_int=char_to_int,
                                              partitions=partitions,
                                              test=test)
    else:
        # Character level encoding
        character_set = settings.CHARACTER_SET
        char_to_int = convert_to_int(sorted(character_set))
        generate_character_level_input_target_data(transcripts=transcripts,
                                                   num_partition=dataset_number,
                                                   char_to_int=char_to_int,
                                                   partitions=partitions,
                                                   test=test)
