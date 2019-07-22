from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from etc import settings

from .encoder_decoder import get_encoder_states, get_encoder_states_gpu, get_decoder_outputs, get_decoder_outputs_gpu

from .encoder_decoder import encoder_bi_GRU, encoder_bi_GRU_gpu, decoder_for_bidirectional_encoder_GRU, \
    decoder_for_bidirectional_encoder_GRU_gpu


def train_baseline_seq2seq_model(mfcc_features, target_length, latent_dim, word_level, gpu_enabled=False):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    pre_decoder_dense_layer = Dense(44, activation="relu", name="pre_decoder_dense")
    decoder_entries = pre_decoder_dense_layer(decoder_inputs)
    dropout_layer = Dropout(0.1, name="decoder_dropout")
    decoder_entries = dropout_layer(decoder_entries)

    if gpu_enabled:

        # Encoder model
        encoder_states = get_encoder_states_gpu(input_shape=mfcc_features,
                                                encoder_inputs=encoder_inputs,
                                                latent_dim=latent_dim)

        # Decoder training, using 'encoder_states' as initial state.
        decoder_inputs = Input(shape=(None, target_length), name="decoder_input")

        decoder_outputs = get_decoder_outputs_gpu(target_length=target_length,
                                                  encoder_states=encoder_states,
                                                  decoder_inputs=decoder_entries,
                                                  latent_dim=latent_dim)
    else:
        # Encoder model
        encoder_states = get_encoder_states(input_shape=mfcc_features,
                                            encoder_inputs=encoder_inputs,
                                            latent_dim=latent_dim)

        # Decoder training, using 'encoder_states' as initial state.
        decoder_outputs = get_decoder_outputs(target_length=target_length,
                                              encoder_states=encoder_states,
                                              decoder_inputs=decoder_entries,
                                              latent_dim=latent_dim)

    # Dense Output Layers
    if word_level:
        target_length = len(settings.CHARACTER_SET) + 1
        decoder_outputs = get_multi_output_dense(decoder_outputs, target_length)
    else:
        decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


def train_bidirectional_baseline_seq2seq_model(mfcc_features, target_length, latent_dim, word_level, gpu_enabled=False):
    """
        trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
        :param mfcc_features: int
        :param target_length: int
        :param latent_dim: int
        :return: Model, Model, Model
        """
    # Encoder training

    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")

    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")

    pre_decoder_dense_layer = Dense(44, activation="relu", name="pre_decoder_dense")
    decoder_entries = pre_decoder_dense_layer(decoder_inputs)
    dropout_layer = Dropout(0.1, name="decoder_dropout")
    decoder_entries = dropout_layer(decoder_entries)

    if gpu_enabled:
        encoder_states = encoder_bi_GRU_gpu(input_shape=mfcc_features,
                                        encoder_inputs=encoder_inputs,
                                        latent_dim=latent_dim)

        # Decoder training, using 'encoder_states' as initial state.


        decoder_outputs = decoder_for_bidirectional_encoder_GRU_gpu(target_length=target_length,
                                                                                encoder_states=encoder_states,
                                                                                decoder_inputs=decoder_entries,
                                                                                latent_dim=latent_dim)

    else:
        encoder_states = encoder_bi_GRU(input_shape=mfcc_features,
                                            encoder_inputs=encoder_inputs,
                                            latent_dim=latent_dim)

        # Decoder training, using 'encoder_states' as initial state.

        decoder_outputs = decoder_for_bidirectional_encoder_GRU(target_length=target_length,
                                                                    encoder_states=encoder_states,
                                                                    decoder_inputs=decoder_entries,
                                                                    latent_dim=latent_dim)


    # Dense Output Layers
    if word_level:
        target_length = len(settings.CHARACTER_SET) + 1
        decoder_outputs = get_multi_output_dense(decoder_outputs, target_length)
    else:
        decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

    # Generating Keras Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


def get_multi_output_dense(decoder_outputs, target_length):
    dense_layers = []

    for i in range(0, settings.LONGEST_WORD_LENGTH):
        decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense" + str(i))
        new_decoder_output = decoder_dense(decoder_outputs)
        dense_layers.append(new_decoder_output)
    return dense_layers
