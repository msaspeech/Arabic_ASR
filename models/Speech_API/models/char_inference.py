from models.Speech_API.utils import load_pickle_data, buckwalter_to_arabic
from models.Speech_API.etc import settings
from tensorflow.python.keras import models
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Input
from models.Speech_API.utils import convert_to_int, convert_int_to_char
import numpy as np
from .word_correction import correct_word
 
class Char_Inference():

    def __init__(self, architecture=0, latent_dim=300):

        model_name = "architecture" + str(architecture)

        model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "char.h5"

        self.model = models.load_model(model_path)
        self.encoder_states = None
        self.latent_dim = latent_dim
        self.encoder_model = None
        self.decoder_model = None

        if architecture == 1:
            self._get_encoder_decoder_model_baseline()
        else:
            self._get_encoder_decoder_model_cnn()

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
        pre_decoder_dense_layer = self.model.get_layer("pre_decoder_dense")
        decoder_dropout = self.model.get_layer("decoder_dropout")

        decoder_gru1_layer = self.model.get_layer("decoder_gru1_layer")
        decoder_gru2_layer = self.model.get_layer("decoder_gru2_layer")
        decoder_dense_layer = self.model.get_layer("decoder_dense")

        decoder_state_input_h1 = Input(shape=(self.latent_dim,))
        decoder_state_input_h2 = Input(shape=(self.latent_dim,))

        decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_h2]

        decoder_entries = pre_decoder_dense_layer(decoder_inputs)
        decoder_entries = decoder_dropout(decoder_entries)

        decoder_gru1, state_h1 = decoder_gru1_layer(decoder_entries, initial_state=decoder_state_input_h1)
        decoder_output, state_h2 = decoder_gru2_layer(decoder_gru1, initial_state=decoder_state_input_h2)

        decoder_states = [state_h1, state_h2]

        # getting dense layers as outputs
        decoder_output = decoder_dropout(decoder_output)
        decoder_output = decoder_dense_layer(decoder_output)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_output] + decoder_states)

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

        pre_decoder_dense_layer = self.model.get_layer("pre_decoder_dense")
        decoder_dropout = self.model.get_layer("decoder_dropout")

        decoder_gru1_layer = self.model.get_layer("decoder_gru1_layer")
        decoder_gru2_layer = self.model.get_layer("decoder_gru2_layer")

        decoder_dense_layer = self.model.get_layer("decoder_dense")

        decoder_state_input_h1 = Input(shape=(self.latent_dim,))
        decoder_state_input_h2 = Input(shape=(self.latent_dim,))

        decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_h2]

        decoder_entries = pre_decoder_dense_layer(decoder_inputs)
        decoder_entries = decoder_dropout(decoder_entries)

        decoder_gru1, state_h1 = decoder_gru1_layer(decoder_entries, initial_state=decoder_state_input_h1)
        decoder_output, state_h2 = decoder_gru2_layer(decoder_gru1, initial_state=decoder_state_input_h2)

        decoder_states = [state_h1, state_h2]

        # getting dense layers as outputs
        decoder_output = decoder_dropout(decoder_output)
        decoder_output = decoder_dense_layer(decoder_output)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_output] + decoder_states)

        self.decoder_model.summary()

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
        zeros2 = np.zeros((1,self.latent_dim))
        states_value = [states_value, zeros2]
        print(zeros2.shape)
        num_decoder_tokens = len(char_to_int)
        target_sequence = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        target_sequence[0, 0, char_to_int['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        max_length = 30
        i = 0
        while not stop_condition:
            outputs = self.decoder_model.predict(
                [target_sequence] + states_value)
            output_tokens = outputs[0]
            states_value = outputs[1:]

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = int_to_char[sampled_token_index]
            decoded_sentence += sampled_char
            if sampled_char == "\n" or len(decoded_sentence) > max_length :
                # End of transcription
                stop_condition = True
            else:
                # updating target sequence vector
                target_sequence = np.zeros((1, 1, num_decoder_tokens))
                target_sequence[0, 0, char_to_int[sampled_char]] = 1
                i += 1

        corrected_sentence = []
        words = decoded_sentence.split()
        for word in words:
            corrected_word = correct_word(word)
            corrected_sentence.append(corrected_word)

        sentence = " ".join(corrected_sentence)

        return buckwalter_to_arabic(sentence)