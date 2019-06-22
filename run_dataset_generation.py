from data import generate_dataset

# from etc import RAW_DATASET_AUDIO_PATH, RAW_DATASET_TRANSCRIPTIONS
from data import generate_pickle_dataset

#generate_dataset()

generate_pickle_dataset(threshold=10)
