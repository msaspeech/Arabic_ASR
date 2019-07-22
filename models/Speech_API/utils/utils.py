import json
import os
import pickle
import re
import socket


def read_file_content(file_path):
    file_lines = []
    with open(file_path, "r") as file:
        file_lines = file.readlines()

    return file_lines


def get_files(directory):
    list_files = []
    for root, dirs, files in os.walk(directory):
        for filename in sorted(files):
            list_files.append(directory + filename)
        list_files.sort(key=natural_keys)
    return list_files


def get_files_full_path(directory):
    files_to_return = []
    for path, subdirs, files in os.walk(directory):
        for name in sorted(files):
            file_path = os.path.join(path, name)
            files_to_return.append(file_path)
    files_to_return.sort(key=natural_keys)
    return files_to_return


def file_exists(file_path):
    if os.path.exists(file_path):
        return True
    return False


def empty_directory(directory_path):
    if not os.listdir(directory_path):
        return True
    return False


def create_dir(dir_path):
    os.mkdir(dir_path)


def get_longest_word_length(words_list):
    max_length = 0
    for word in words_list:
        if len(word) > max_length:
            max_length = len(word)

    return max_length


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def int_to_binary(num):
    if num == 0:
        return [num]

    binary_digits = []
    while num != 0:
        modnum = int(num % 2)
        num = int(num / 2)
        binary_digits.append(modnum)
    return list(reversed(binary_digits))


def convert_word_to_binary(word_index, output_binary_vector):
    binary_value = int_to_binary(word_index)
    input_length = len(binary_value) - 1
    output_length = len(output_binary_vector) - 1

    for i in range(0, len(binary_value)):
        # for i, value in enumerate(binary_value):
        output_binary_vector[output_length - i] = binary_value[input_length - i]

    return output_binary_vector


def get_empty_binary_vector(upper_bound):
    binary_vector = int_to_binary(upper_bound)
    for i in range(0, len(binary_vector)):
        binary_vector[i] = 0
    return binary_vector


def generate_pickle_file(data, file_path):
    """
     Uploads AudioInput data from pickle file
     :return:
     """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def load_pickle_data(file_path):
    """
         Uploads AudioInput data after padding from pickle file
         :return:
         """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def generate_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data



def is_connected():
  REMOTE_SERVER = "www.google.com"
  try:
    # see if we can resolve the host name -- tells us if there is
    # a DNS listening
    host = socket.gethostbyname(REMOTE_SERVER)
    # connect to the host -- tells us if the host is actually
    # reachable
    s = socket.create_connection((host, 80), 2)
    s.close()
    return True
  except:
     pass
  return False