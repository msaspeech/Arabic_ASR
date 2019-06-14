import sys

from data import upload_dataset_partition
from etc import DRIVE_INSTANCE_PATH
from etc import settings
from init_directories import init_directories
from models import Seq2SeqModel
from utils import load_pickle_data

init_directories()
settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)

word_level = int(sys.argv[1])
architecture = int(sys.argv[2])
latent_dim = int(sys.argv[3])
epochs = int(sys.argv[4])

upload_dataset_partition(word_level=word_level, partitions=64)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
model.train_model()

# accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
