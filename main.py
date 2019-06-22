import sys
from data import upload_dataset_partition
from init_directories import init_directories
from models import Seq2SeqModel

init_directories()

word_level = int(sys.argv[1])
architecture = int(sys.argv[2])
latent_dim = int(sys.argv[3])
epochs = int(sys.argv[4])

upload_dataset_partition(word_level=word_level, partitions=64)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
model.train_model()

