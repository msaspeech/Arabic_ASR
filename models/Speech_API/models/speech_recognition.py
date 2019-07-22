from models.Speech_API.utils import load_pickle_data
from models.Speech_API.etc import settings
from models.Speech_API.models.audio_input import AudioInput
import numpy as np
from .char_inference import Char_Inference

def recognize_speech(audio_sample, latent_dim=300, architecture=0, word_level=0):

    if word_level :
        general_info = load_pickle_data(settings.DATASET_WORD_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.TOTAL_SAMPLES_NUMBER = general_info[1]
        settings.WORD_SET = general_info[2]
        settings.LONGEST_WORD_LENGTH = general_info[3]
        settings.CHARACTER_SET = general_info[4]
        settings.WORD_TARGET_LENGTH = general_info[5]

        sample = AudioInput(audio_sample, "")
        audio = [sample.mfcc.transpose()]

        audio_sequence = np.array(audio, dtype=np.float32)
        return None

    else:

        general_info = load_pickle_data(settings.DATASET_CHAR_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.CHARACTER_SET = general_info[2]
        general_info = load_pickle_data(settings.DATASET_WORD_INFORMATION_PATH)
        settings.WORD_SET = general_info[2]

        sample = AudioInput(audio_sample, "")
        audio = [sample.mfcc.transpose()]

        audio_sequence = np.array(audio, dtype=np.float32)

        char_inference = Char_Inference(architecture, latent_dim)

        decoded_sentence = char_inference.decode_audio_sequence_character_based(audio_sequence)

        return decoded_sentence
