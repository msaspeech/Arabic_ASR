import os
from utils import file_exists
from etc import settings


def init_directories():
    list_paths = [settings.DATA_PATH, settings.GENERATED_DATA_PATH, settings.PICKLE_PARTITIONS_PATH, settings.GENERATED_DATA_WAV_PATH,
                  settings.GENERATED_DATA_TRANSCRIPTS_PATH, settings.DATASET_SPLIT_PATH, settings.DATASET_CHAR, settings.DATASET_WORD,
                  settings.NORMALIZATION_PATH, settings.DATASET_SPLIT_WORD_TRAIN_PATH,
                  settings.DATASET_SPLIT_WORD_TEST_PATH, settings.DATASET_SPLIT_CHAR_TRAIN_PATH,
                  settings.DATASET_SPLIT_CHAR_TEST_PATH, settings.AUDIO_WORD_SPLIT_TRAIN_PATH,
                  settings.AUDIO_CHAR_SPLIT_TRAIN_PATH, settings.AUDIO_CHAR_SPLIT_TEST_PATH,
                  settings.AUDIO_WORD_SPLIT_TEST_PATH, settings.TRANSCRIPTS_ENCODING_CHAR_SPLIT_TRAIN_PATH,
                  settings.TRANSCRIPTS_ENCODING_CHAR_SPLIT_TEST_PATH, settings.TRANSCRIPTS_ENCODING_WORD_SPLIT_TRAIN_PATH,
                  settings.TRANSCRIPTS_ENCODING_WORD_SPLIT_TEST_PATH, settings.TRAINED_MODELS_PATH]


    for path in list_paths:
        if not file_exists(path):
            os.mkdir(path)



