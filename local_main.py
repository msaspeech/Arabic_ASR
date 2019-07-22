from data import upload_dataset_partition
from init_directories import init_directories
from models import Seq2SeqModel

init_directories()

architecture = 5
word_level = 1
latent_dim = 350
epochs = 500

upload_dataset_partition(word_level=word_level, partitions=2)
model = Seq2SeqModel(latent_dim=latent_dim, epochs=epochs, model_architecture=architecture, word_level=word_level)
model.train_model()

