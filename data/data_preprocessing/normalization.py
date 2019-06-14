import gc

import numpy as np

from etc import settings
from utils import generate_pickle_file


def get_attribute_values(dataset, attribute_index):
    attribute_values = []
    for sample in dataset:
        elements = sample[:, attribute_index]
        for elt in elements:
            attribute_values.append(elt)

    return attribute_values


def min_max_normalization(value, index_column, min_attributes, max_attributes):
    normalized = (value - min_attributes[index_column]) / (max_attributes[index_column] - min_attributes[index_column])
    return normalized


def _group_splitted_datasets(splitted_dataset):
    dataset = []
    for d in splitted_dataset:
        dataset = dataset + d
    return d


def normalize_encoder_input(dataset):
    first_interval = int(len(dataset) / 4)
    second_interval = int(len(dataset) / 2)
    third_interval = int(len(dataset) * 3 / 4)

    splitted_dataset = []
    splitted_dataset.append(dataset[0: first_interval])
    splitted_dataset.append(dataset[first_interval: second_interval])
    splitted_dataset.append(dataset[second_interval: third_interval])
    splitted_dataset.append(dataset[third_interval: len(dataset)])

    min_attributes = []
    max_attributes = []
    print("into normalization")
    # Calculating and saving min and max values of dataset
    for attribute_index in range(0, settings.MFCC_FEATURES_LENGTH):
        attribute_values = get_attribute_values(dataset, attribute_index)
        min_attributes.append(np.min(attribute_values))
        max_attributes.append(np.max(attribute_values))

    print(min_attributes)
    print(max_attributes)
    generate_pickle_file(min_attributes, settings.ENCODER_INPUT_MIN_VALUES_PATH)
    generate_pickle_file(max_attributes, settings.ENCODER_INPUT_MAX_VALUES_PATH)

    print("generating new dataset")

    del dataset
    gc.collect()
    dataset_index = 0
    while splitted_dataset:
        for i, encoder_input in enumerate(splitted_dataset[0]):
            normalized_encoder_input = []
            for line in encoder_input:
                normalized_line = []
                for index_column, value in enumerate(line):
                    normalized_line.append(min_max_normalization(value, index_column, min_attributes, max_attributes))
                normalized_encoder_input.append(normalized_line)
            print("normalized " + str(i))
            splitted_dataset[0][i] = normalized_encoder_input
        generate_pickle_file(splitted_dataset[0],
                             settings.NORMALIZED_ENCODER_INPUT_PATHS + "dataset" + str(dataset_index) + ".pkl")

        splitted_dataset.pop(0)
        gc.collect()
        dataset_index += 1

    final_dataset = _group_splitted_datasets(splitted_dataset)

    return final_dataset

    # for i, encoder_input in enumerate(dataset):
    #    normalized_encoder_input = []
    #    for line in encoder_input:
    #        normalized_line = []
    #        for index_column, value in enumerate(line):
    #            normalized_line.append(min_max_normalization(value, index_column, min_attributes, max_attributes))
    #        normalized_encoder_input.append(normalized_line)
    #    print("normalized "+str(i))
    #    dataset[i] = normalized_encoder_input

    # generate_pickle_file(dataset, settings.NORMALIZED_ENCODER_INPUT_PATH)
    # return dataset
