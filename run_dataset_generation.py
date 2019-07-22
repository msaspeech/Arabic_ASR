from data import generate_dataset
from init_directories import init_directories
from data import generate_pickle_dataset_xml

init_directories()

generate_dataset()

generate_pickle_dataset_xml(threshold=0.5)
