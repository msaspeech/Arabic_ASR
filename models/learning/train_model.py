import random

import numpy as np
from tensorflow.python.keras import models

from etc import settings
from utils import file_exists, get_files, load_pickle_data, get_files_full_path
from .model_callback import ModelSaver
from .seq2seq_baseline import train_baseline_seq2seq_model_GRU, train_bidirectional_baseline_seq2seq_model_GRU
from .seq2seq_cnn_attention import train_cnn_seq2seq_model_GRU, train_cnn_bidirectional_seq2seq_model_GRU


class Seq2SeqModel():
    def __init__(self, latent_dim=300, epochs=50, model_architecture=5, data_generation=True, word_level=False):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.model_architecture = model_architecture
        self.data_generation = data_generation
        self.model_name = "architecture" + str(self.model_architecture)
        self.mfcc_features_length = settings.MFCC_FEATURES_LENGTH
        self.word_level = word_level
        if word_level:
            self.model_path = settings.TRAINED_MODELS_PATH + self.model_name + "/" + self.model_name + "word.h5"
            self.target_length = settings.WORD_TARGET_LENGTH
        else:
            self.model_path = settings.TRAINED_MODELS_PATH + self.model_name + "/" + self.model_name + "char.h5"
            self.target_length = len(settings.CHARACTER_SET)

        self.model = None
        self.encoder_states = None
        self._load_model()

    def test_model(self):

        if file_exists(self.model_path):
            self.model = models.load_model(self.model_path)

            test_samples = int(settings.TOTAL_SAMPLES_NUMBER * 5 / 100)
            if self.word_level:
                self.model.evaluate_generator(self.split_data_generator_dict_word_level_test(), steps=test_samples,
                                              verbose=1)
            else:
                self.model.evaluate_generator(self.split_data_generator_dict_test(), test_samples, verbose=1)

    def _load_model(self):
        if file_exists(self.model_path):
            self.model = models.load_model(self.model_path)
            print(self.model.summary())
        else:
            if self.model_architecture == 1:
                print("BASELINE MODEL")
                self.model, self.encoder_states = train_baseline_seq2seq_model_GRU(
                    mfcc_features=self.mfcc_features_length,
                    target_length=self.target_length,
                    latent_dim=self.latent_dim,
                    word_level=self.word_level)

            elif self.model_architecture == 2:
                print("BI BASELINE MODEL")
                self.model, self.encoder_states = train_bidirectional_baseline_seq2seq_model_GRU(
                    mfcc_features=self.mfcc_features_length,
                    target_length=self.target_length,
                    latent_dim=self.latent_dim,
                    word_level=self.word_level)

            elif self.model_architecture == 3:
                print("BASELINE MODEL")
                self.model, self.encoder_states = train_baseline_seq2seq_model_GRU(
                    mfcc_features=self.mfcc_features_length,
                    target_length=self.target_length,
                    latent_dim=self.latent_dim,
                    word_level=self.word_level)


            elif self.model_architecture == 4:
                print("CNN MODEL")
                self.model, self.encoder_states = train_cnn_seq2seq_model_GRU(mfcc_features=self.mfcc_features_length,
                                                                              target_length=self.target_length,
                                                                              latent_dim=self.latent_dim,
                                                                              word_based=self.word_level)

            elif self.model_architecture == 5:
                print("CNN MODEL")
                self.model, self.encoder_states = train_cnn_seq2seq_model_GRU(mfcc_features=self.mfcc_features_length,
                                                                              target_length=self.target_length,
                                                                              latent_dim=self.latent_dim,
                                                                              word_based=self.word_level)
            elif self.model_architecture == 6:
                print("BASELINE MODEL")
                self.model, self.encoder_states = train_baseline_seq2seq_model_GRU(
                    mfcc_features=self.mfcc_features_length,
                    target_length=self.target_length,
                    latent_dim=self.latent_dim,
                    word_level=self.word_level)

            else:
                print("BI CNN MODEL")
                self.model, self.encoder_states = train_cnn_bidirectional_seq2seq_model_GRU(
                    mfcc_features=self.mfcc_features_length,
                    target_length=self.target_length,
                    latent_dim=self.latent_dim,
                    word_level=self.word_level)

    def train_model(self):
        print("ENCODER STATES")

        model_saver = ModelSaver(model_name=self.model_name, model_path=self.model_path,
                                 encoder_states=self.encoder_states,
                                 drive_instance=settings.DRIVE_INSTANCE,
                                 word_level=self.word_level)

        if self.word_level:
            loss = dict()
            for i in range(0, settings.LONGEST_WORD_LENGTH):
                layer_name = "decoder_dense" + str(i)
                loss[layer_name] = 'categorical_crossentropy'

            self.model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy'])
            batch_size = 32
            steps = int(settings.TOTAL_SAMPLES_NUMBER / batch_size) + 1
            history = self.model.fit_generator(self.split_data_generator_dict_word_level(batch_size),
                                               steps_per_epoch=steps,
                                               epochs=self.epochs,
                                               callbacks=[model_saver])
        else:
            print("training here")
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            batch_size = 32
            # steps = int(settings.TOTAL_SAMPLES_NUMBER / batch_size) + 1
            history = self.model.fit_generator(self.split_data_generator_dict_train(),
                                               steps_per_epoch=30000,
                                               epochs=self.epochs,
                                               callbacks=[model_saver])

    def validation_generator(self):
        audio_directory = settings.AUDIO_SPLIT_TEST_PATH
        audio_files = get_files(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH
        transcript_files = get_files(transcripts_directory)
        data = self.get_data(audio_files[0], transcript_files[0])

        while True:
            pair_key = random.choice(list(data.keys()))
            output = data[pair_key]
            encoder_x = []
            decoder_x = []
            decoder_y = []
            for element in output:
                encoder_x.append(element[0][0])
                decoder_x.append(element[0][1])
                decoder_y.append(element[1])

            encoder_x = np.array(encoder_x)
            decoder_x = np.array(decoder_x)
            decoder_y = np.array(decoder_y)

            yield [encoder_x, decoder_x], decoder_y

    def split_data_generator_dict_test(self):
        audio_directory = settings.AUDIO_SPLIT_TEST_PATH
        audio_files = get_files_full_path(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH
        transcript_files = get_files_full_path(transcripts_directory)
        while True:
            for i, audio_file in enumerate(audio_files):
                # retrieving data

                data = self.get_data(audio_file, transcript_files[i])

                for key_pair in data:
                    output = data[key_pair]
                    encoder_x = []
                    decoder_x = []
                    decoder_y = []
                    for element in output:
                        encoder_x.append(element[0][0])
                        decoder_x.append(element[0][1])
                        decoder_y.append(element[1])

                    encoder_x = np.array(encoder_x)
                    decoder_x = np.array(decoder_x)
                    decoder_y = np.array(decoder_y)
                    yield [encoder_x, decoder_x], decoder_y

    def split_data_generator_dict_train(self):
        audio_directory = settings.AUDIO_SPLIT_TRAIN_PATH
        audio_files = get_files_full_path(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH
        transcript_files = get_files_full_path(transcripts_directory)
        while True:
            for i, audio_file in enumerate(audio_files):
                # retrieving data

                data = self.get_data(audio_file, transcript_files[i])

                for key_pair in data:
                    output = data[key_pair]
                    encoder_x = []
                    decoder_x = []
                    decoder_y = []
                    for element in output:
                        encoder_x.append(element[0][0])
                        decoder_x.append(element[0][1])
                        decoder_y.append(element[1])

                    encoder_x = np.array(encoder_x)
                    decoder_x = np.array(decoder_x)
                    decoder_y = np.array(decoder_y)
                    yield [encoder_x, decoder_x], decoder_y

    def split_data_generator_dict(self, batch_size):
        audio_directory = settings.AUDIO_SPLIT_TRAIN_PATH
        audio_files = get_files_full_path(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH
        transcript_files = get_files_full_path(transcripts_directory)
        while True:
            for i, audio_file in enumerate(audio_files):
                # retrieving data

                data = self.get_data(audio_file, transcript_files[i])
                size = sum(len(d) for d in data.values())
                probas = dict((d, len(data[d]) / size) for d in data)
                keys = sorted(data.keys())
                loop_size = int(size / batch_size) + 1
                for i in range(loop_size):
                    r = random.random()
                    for key in keys:
                        if r < probas[key]:
                            break
                        r -= probas[key]
                    output = data[key]
                    b_size = min((batch_size, len(output)))
                    output = random.sample(output, b_size)
                    encoder_x = []

                    decoder_x = []
                    decoder_y = []
                    for element in output:
                        encoder_x.append(element[0][0])
                        decoder_x.append(element[0][1])
                        decoder_y.append(element[1])

                    encoder_x = np.array(encoder_x)
                    decoder_x = np.array(decoder_x)
                    decoder_y = np.array(decoder_y)
                    yield [encoder_x, decoder_x], decoder_y

    def split_data_generator_dict_word_level(self, batch_size):
        audio_directory = settings.AUDIO_SPLIT_TRAIN_PATH
        audio_files = get_files_full_path(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TRAIN_PATH
        transcript_files = get_files_full_path(transcripts_directory)

        while True:
            for i, audio_file in enumerate(audio_files):
                # retrieving data
                data = self.get_data(audio_file, transcript_files[i])
                size = sum(len(d) for d in data.values())
                probas = dict((d, len(data[d]) / size) for d in data)
                keys = sorted(data.keys())
                loop_size = int(size / batch_size) + 1
                for i in range(loop_size):
                    r = random.random()
                    for key in keys:
                        if r < probas[key]:
                            break
                        r -= probas[key]
                    output = data[key]
                    b_size = min((batch_size, len(output)))
                    output = random.sample(output, b_size)
                    encoder_x = []
                    decoder_x = []
                    decoder_y = []
                    for element in output:
                        encoder_x.append(element[0][0])
                        decoder_x.append(element[0][1])
                        decoder_y.append(element[1])

                    encoder_x = np.array(encoder_x)
                    decoder_x = np.array(decoder_x)

                    # decoder_target = decoder_y.copy()

                    decoder_target = []
                    for h in range(0, len(decoder_y)):
                        decoder_target.append(decoder_y[h].copy())

                    decoder_targets = []
                    num_words = len(decoder_y[h])
                    for j in range(0, settings.LONGEST_WORD_LENGTH):
                        for i in range(0, num_words):
                            for h in range(0, len(decoder_y)):
                                decoder_target[h][i] = decoder_y[h][i][j]
                        decoder_targets.append(np.array(decoder_target))

                    yield [encoder_x, decoder_x], decoder_targets

    def split_data_generator_dict_word_level_test(self):
        audio_directory = settings.AUDIO_SPLIT_TEST_PATH
        audio_files = get_files_full_path(audio_directory)
        transcripts_directory = settings.TRANSCRIPTS_ENCODING_SPLIT_TEST_PATH_PARTITION
        transcript_files = get_files_full_path(transcripts_directory)

        while True:
            for i, audio_file in enumerate(audio_files):
                # retrieving data
                data = self.get_data(audio_file, transcript_files[i])
                for key_pair in data:
                    output = data[key_pair]
                    encoder_x = []
                    decoder_x = []
                    decoder_y = []
                    for element in output:
                        encoder_x.append(element[0][0])
                        decoder_x.append(element[0][1])
                        decoder_y.append(element[1])

                    encoder_x = np.array(encoder_x)
                    decoder_x = np.array(decoder_x)

                    # decoder_target = decoder_y.copy()

                    decoder_target = []
                    for h in range(0, len(decoder_y)):
                        decoder_target.append(decoder_y[h].copy())

                    decoder_targets = []
                    num_words = len(decoder_y[h])
                    for j in range(0, settings.LONGEST_WORD_LENGTH):
                        for i in range(0, num_words):
                            for h in range(0, len(decoder_y)):
                                decoder_target[h][i] = decoder_y[h][i][j]
                        decoder_targets.append(np.array(decoder_target))

                    yield [encoder_x, decoder_x], decoder_targets

    def get_test_data(self, audio_file, transcripts_file):
        encoder_input_data = load_pickle_data(audio_file)
        (decoder_input_data, decoder_target_data) = load_pickle_data(transcripts_file)
        return encoder_input_data, decoder_input_data, decoder_target_data

    def get_data(self, audio_file, transcripts_file):
        encoder_input_data = load_pickle_data(audio_file)
        (decoder_input_data, decoder_target_data) = load_pickle_data(transcripts_file)
        data = self._generate_timestep_dict(encoder_input_data, decoder_input_data, decoder_target_data)
        return data

    def _generate_timestep_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
        generated_data = dict()
        for index, encoder_input in enumerate(encoder_input_data):
            key_pair = (len(encoder_input), len(decoder_input_data[index]))
            if not key_pair in generated_data:
                generated_data[key_pair] = []
            generated_data[key_pair].append([[encoder_input, decoder_input_data[index]], decoder_target_data[index]])

        return generated_data
