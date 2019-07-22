import sys
from data import upload_dataset_partition
from init_directories import init_directories
from models import Seq2SeqModel

init_directories()

word_level = 0
architecture = 1
latent_dim = 300
epochs = 10

upload_dataset_partition(word_level=word_level, partitions=64)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
model.train_model()

