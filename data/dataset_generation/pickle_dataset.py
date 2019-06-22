import gc

from .audio_transcript_map import map_audio_transcripts_xml, map_audio_transcripts_generic
from ..data_preprocessing.transcript_preprocessing import transcript_preprocessing, special_characters_table
from etc import settings
from lib import AudioInput
from utils import generate_pickle_file


def generate_pickle_dataset(threshold=10):
    """
    Generates pickle dataset_out of transcription and
    :param threshold:
    :return:
    """

    threshold = threshold * 3600
    mapped_audio = map_audio_transcripts_generic()
    audioInput = []
    timing = 0
    pickle_file_index = 0
    for path, transcription in mapped_audio.items():
        print(path + "====>" + transcription)

        # Calling transcription preprocessing
        special_characters = special_characters_table()
        transcription = transcript_preprocessing(transcription, special_characters)

        if transcription is not None:
            # Calculating Total audio length for partitions
            audioInput_instance = AudioInput(path=path, transcript=transcription)
            audioInput.append(audioInput_instance)

            timing += audioInput_instance.audio_length
            print("Timing is : " + str(timing))

        if timing >= threshold:
            path = settings.PICKLE_PARTITIONS_PATH + "dataset" + str(pickle_file_index) + ".pkl"
            generate_pickle_file(audioInput, path)
            pickle_file_index += 1
            timing = 0
            del audioInput
            gc.collect()
            audioInput = []


def generate_pickle_dataset_xml(threshold):
    """
    Generates pickle dataset_out of transcription and
    :param threshold:
    :return:
    """

    threshold = threshold * 3600
    mapped_audio = map_audio_transcripts_xml()
    audioInput = []
    timing = 0
    pickle_file_index = 0
    for path, transcription in mapped_audio.items():
        print(path + "====>" + transcription)

        # Calling transcription preprocessing
        special_characters = special_characters_table()
        transcription = transcript_preprocessing(transcription, special_characters)

        if transcription is not None:
            # Calculating Total audio length for partitions
            audioInput_instance = AudioInput(path=path, transcript=transcription)
            audioInput.append(audioInput_instance)

            timing += audioInput_instance.audio_length
            print("Timing is : " + str(timing))

        if timing >= threshold:
            path = settings.PICKLE_PARTITIONS_PATH + "dataset" + str(pickle_file_index) + ".pkl"
            generate_pickle_file(audioInput, path)
            pickle_file_index += 1
            timing = 0
            del audioInput
            gc.collect()
            audioInput = []
