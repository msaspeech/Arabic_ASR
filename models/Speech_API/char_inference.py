from etc import settings
from utils import load_pickle_data
from tensorflow.python.keras import models
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Input
from utils import convert_to_int, convert_int_to_char
import numpy as np

class Char_Inference():

    def __init__(self, word_level, architecture, latent_dim):

        model_name = "architecture" + str(architecture)
        if word_level:
            model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "word.h5"
        else:
            model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "char.h5"

        print(model_path)

        self.model = models.load_model(model_path)
        self.encoder_states = None
        self.latent_dim = latent_dim
        general_info = load_pickle_data(settings.DATASET_CHAR_INFERENCE_INFORMATION_PATH)
        settings.MFCC_FEATURES_LENGTH = general_info[0]
        settings.CHARACTER_SET = general_info[2]
        self.encoder_model = None
        self.decoder_model = None


        if architecture == 6:
            self._get_encoder_decoder_model_baseline()
        else:
            self._get_encoder_decoder_model_cnn()

    def predict_sequence_test(self, audio_input):
        char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
        int_to_char = convert_int_to_char(char_to_int)

        t_force = "\tmsA' Alxyr >wlA hw Almwqf tm tSHyHh\n"
        encoded_transcript = []
        for index, character in enumerate(t_force):
            encoded_character = [0] * len(settings.CHARACTER_SET)
            position = char_to_int[character]
            encoded_character[position] = 1
            encoded_transcript.append(encoded_character)

        decoder_input = np.array([encoded_transcript])
        print(decoder_input.shape)

        output = self.model.predict([audio_input, decoder_input])
        print(output.shape)
        sentence = ""
        output = output[0]
        for character in output:
            position = np.argmax(character)
            character = int_to_char[position]
            sentence+=character

        print(sentence)

    def _get_encoder_decoder_model_cnn(self):
        # Getting encoder model
        encoder_inputs = self.model.get_layer("encoder_input").input

        cnn_model = self.model.get_layer("sequential")
        encoder_inputs_cnn = cnn_model(encoder_inputs)

        encoder_gru = self.model.get_layer("encoder_gru_layer")
        encoder_output, h = encoder_gru(encoder_inputs_cnn)
        self.encoder_states = h

        self.encoder_model = Model(encoder_inputs, self.encoder_states)
        self.encoder_model.summary()
        # Getting decoder model

        decoder_inputs = self.model.get_layer("decoder_input").input

        decoder_gru1_layer = self.model.get_layer("decoder_gru1_layer")
        decoder_gru2_layer = self.model.get_layer("decoder_gru2_layer")
        decoder_gru3_layer = self.model.get_layer("decoder_gru3_layer")

        decoder_dropout = self.model.get_layer("decoder_dropout")
        decoder_dense_layer = self.model.get_layer("decoder_dense")

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h]

        decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_gru2 = decoder_gru2_layer(decoder_gru1)
        decoder_output = decoder_gru3_layer(decoder_gru2)

        decoder_states = [state_h]

        # getting dense layers as outputs
        decoder_output  = decoder_dropout(decoder_output)
        decoder_output = decoder_dense_layer(decoder_output)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_output] + decoder_states)

    def _get_encoder_decoder_model_baseline(self):

        # Getting encoder model
        encoder_inputs = self.model.get_layer("encoder_input").input
        encoder_gru = self.model.get_layer("encoder_gru_layer")
        encoder_output, h = encoder_gru(encoder_inputs)
        self.encoder_states = h

        self.encoder_model = Model(encoder_inputs, self.encoder_states)
        self.encoder_model.summary()
        # Getting decoder model

        decoder_inputs = self.model.get_layer("decoder_input").input

        decoder_gru1_layer = self.model.get_layer("decoder_gru1_layer")
        decoder_gru2_layer = self.model.get_layer("decoder_gru2_layer")
        decoder_gru3_layer = self.model.get_layer("decoder_gru3_layer")


        decoder_dense_layer = self.model.get_layer("decoder_dense")

        decoder_state_input_h = Input(shape=(self.latent_dim, ))
        decoder_states_inputs = [decoder_state_input_h]

        decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_gru2 = decoder_gru2_layer(decoder_gru1)
        decoder_output = decoder_gru3_layer(decoder_gru2)
        decoder_states = [state_h]

        # getting dense layers as outputs

        decoder_output = decoder_dense_layer(decoder_output)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_output] + decoder_states)




    def decode_audio_sequence_character_based(self, audio_sequence):
        """
        Decodes audio sequence into a transcript using encoder_model and decoder_model generated from training
        :param audio_sequence: 2D numpy array
        :param encoder_model: Model
        :param decoder_model: Model
        :param character_set: Dict
        :return: String
        """
        # Getting converters
        char_to_int = convert_to_int(sorted(settings.CHARACTER_SET))
        int_to_char = convert_int_to_char(char_to_int)

        # Returns the encoded audio_sequence
        states_value = self.encoder_model.predict(audio_sequence)
        states_value = [states_value]
        print("ENCODER PREDICTION DONE")
        num_decoder_tokens = len(char_to_int)
        target_sequence = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        target_sequence[0, 0, char_to_int['\t']] = 1.
        print(target_sequence)
        stop_condition = False
        t_force = "\tmsA' Alxyr >wlA hw Almwqf tm tSHyHh\n"
        decoded_sentence = ''
        max_length = len(t_force)
        i = 0
        while not stop_condition:
            output_tokens, h = self.decoder_model.predict(
                [target_sequence] + states_value)
            states_value = [h]
            #print("DECODER PREDICTION DONE khobz")
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = int_to_char[sampled_token_index]
            #print(sampled_char)
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > max_length :
                # End of transcription
                stop_condition = True
            else:
                # updating target sequence vector
                target_sequence = np.zeros((1, 1, num_decoder_tokens))
                target_sequence[0, 0, char_to_int[t_force[i]]] = 1
                i += 1

        print(decoded_sentence)
        return decoded_sentence