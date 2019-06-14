from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input

from etc import settings
from .encoder_decoder import get_encoder_states_GRU, encoder_bi_GRU, \
    decoder_for_bidirectional_encoder_GRU, get_decoder_outputs_GRU_test
from .encoder_decoder import get_encoder_states_LSTM, get_decoder_outputs_LSTM, encoder_bi_LSTM, \
    decoder_for_bidirectional_encoder_LSTM


def train_baseline_seq2seq_model_GRU(mfcc_features, target_length, latent_dim, word_level):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    encoder_states = get_encoder_states_GRU(encoder_inputs=encoder_inputs,
                                            latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    # masked_inputs = Masking(mask_value=0,)(decoder_inputs)

    # decoder_outputs, decoder_states = get_decoder_outputs_GRU(encoder_states=encoder_states,
    #                                                          decoder_inputs=decoder_inputs,
    #                                                          latent_dim=latent_dim)

    decoder_outputs, decoder_states = get_decoder_outputs_GRU_test(encoder_states=encoder_states,
                                                                   decoder_inputs=decoder_inputs,
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
    # print(model.summary())
    return model, encoder_states


def train_bidirectional_baseline_seq2seq_model_GRU(mfcc_features, target_length, latent_dim, word_level):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    encoder_states = encoder_bi_GRU(encoder_inputs=encoder_inputs,
                                    latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.

    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    decoder_outputs, decoder_states = decoder_for_bidirectional_encoder_GRU(encoder_states=encoder_states,
                                                                            decoder_inputs=decoder_inputs,
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
    print(model.summary())
    return model, encoder_states


def train_baseline_seq2seq_model_LSTM(mfcc_features, target_length, latent_dim, word_level):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    encoder_states = get_encoder_states_LSTM(encoder_inputs=encoder_inputs,
                                             latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.
    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    # masked_inputs = Masking(mask_value=0,)(decoder_inputs)
    decoder_outputs, decoder_states = get_decoder_outputs_LSTM(encoder_states=encoder_states,
                                                               decoder_inputs=decoder_inputs,
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
    # print(model.summary())
    return model, encoder_states


def train_bidirectional_baseline_seq2seq_model_LSTM(mfcc_features, target_length, latent_dim, word_level):
    """
    trains Encoder/Decoder architecture and prepares encoder_model and decoder_model for prediction part
    :param mfcc_features: int
    :param target_length: int
    :param latent_dim: int
    :return: Model, Model, Model
    """
    # Encoder training
    encoder_inputs = Input(shape=(None, mfcc_features), name="encoder_input")
    encoder_states = encoder_bi_LSTM(encoder_inputs=encoder_inputs,
                                     latent_dim=latent_dim)

    # Decoder training, using 'encoder_states' as initial state.

    decoder_inputs = Input(shape=(None, target_length), name="decoder_input")
    decoder_outputs, decoder_states = decoder_for_bidirectional_encoder_LSTM(encoder_states=encoder_states,
                                                                             decoder_inputs=decoder_inputs,
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
    print(model.summary())
    return model, encoder_states


def get_multi_output_dense(decoder_outputs, target_length):
    dense_layers = []

    for i in range(0, settings.LONGEST_WORD_LENGTH):
        decoder_dense = Dense(target_length, activation='softmax', name="decoder_dense" + str(i))
        new_decoder_output = decoder_dense(decoder_outputs)
        dense_layers.append(new_decoder_output)
    return dense_layers
