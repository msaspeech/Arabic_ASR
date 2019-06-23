from tensorflow.python.keras.layers import Bidirectional, Concatenate, LSTM, GRU, CuDNNGRU


# from keras.layers import CuDNNLSTM, Bidirectional, Concatenate, LSTM


def get_encoder_states(input_shape, encoder_inputs, latent_dim):
    encoder = GRU(latent_dim,
                  input_shape=(None, input_shape),
                  stateful=False,
                  return_sequences=False,
                  return_state=True,
                  kernel_constraint=None,
                  kernel_regularizer=None,
                  recurrent_initializer='glorot_uniform',
                  name="encoder_gru_layer")
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, state_h = encoder(encoder_inputs)
    encoder_states = [state_h]

    return encoder_states


def get_decoder_outputs(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_gru1_layer = GRU(latent_dim,
                             input_shape=(None, target_length),
                             return_sequences=True,
                             return_state=True,
                             kernel_constraint=None,
                             kernel_regularizer=None,
                             name="decoder_gru1_layer")
    decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_gru2_layer = GRU(latent_dim,
                             stateful=False,
                             return_sequences=True,
                             return_state=False,
                             kernel_constraint=None,
                             kernel_regularizer=None,
                             name="decoder_gru2_layer")
    decoder_gru2 = decoder_gru2_layer(decoder_gru1)

    decoder_gru3_layer = GRU(latent_dim,
                             stateful=False,
                             return_sequences=True,
                             return_state=False,
                             kernel_constraint=None,
                             kernel_regularizer=None,
                             name="decoder_gru3_layer")
    decoder_outputs = decoder_gru3_layer(decoder_gru2)

    return decoder_outputs


def get_encoder_states_gpu(input_shape, encoder_inputs, latent_dim):
    encoder = CuDNNGRU(latent_dim,
                       input_shape=(None, input_shape),
                       stateful=False,
                       return_sequences=False,
                       return_state=True,
                       kernel_constraint=None,
                       kernel_regularizer=None,
                       recurrent_initializer='glorot_uniform',
                       name="encoder_gru_layer")
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, state_h = encoder(encoder_inputs)
    encoder_states = [state_h]

    return encoder_states


def get_decoder_outputs_gpu(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_gru1_layer = CuDNNGRU(latent_dim,
                                  input_shape=(None, target_length),
                                  return_sequences=True,
                                  return_state=True,
                                  kernel_constraint=None,
                                  kernel_regularizer=None,
                                  name="decoder_gru1_layer")
    decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_gru2_layer = CuDNNGRU(latent_dim,
                                  stateful=False,
                                  return_sequences=True,
                                  return_state=False,
                                  kernel_constraint=None,
                                  kernel_regularizer=None,
                                  name="decoder_gru2_layer")
    decoder_gru2 = decoder_gru2_layer(decoder_gru1)

    decoder_gru3_layer = CuDNNGRU(latent_dim,
                                  stateful=False,
                                  return_sequences=True,
                                  return_state=False,
                                  kernel_constraint=None,
                                  kernel_regularizer=None,
                                  name="decoder_gru3_layer")
    decoder_outputs = decoder_gru3_layer(decoder_gru2)

    return decoder_outputs


def encoder_bi_GRU(input_shape, encoder_inputs, latent_dim):
    encoder = Bidirectional(GRU(latent_dim,
                                     input_shape=(None, input_shape),
                                     stateful=False,
                                     return_sequences=False,
                                     return_state=True,
                                     kernel_constraint=None,
                                     kernel_regularizer=None,
                                     recurrent_initializer='glorot_uniform'),
                            name="encoder_bi_gru_layer")
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, forward_h, backward_h = encoder(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    encoder_states = [state_h]

    return encoder_states


def decoder_for_bidirectional_encoder_GRU(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_gru1_layer = GRU(latent_dim * 2,
                                  input_shape=(None, target_length),
                                  return_sequences=True,
                                  return_state=True,
                                  kernel_constraint=None,
                                  kernel_regularizer=None,
                                  name="decoder_bi_gru1_layer")
    decoder_gru1, state_h = decoder_gru1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_gru2_layer = GRU(latent_dim * 2,
                                  stateful=False,
                                  return_sequences=True,
                                  kernel_constraint=None,
                                  kernel_regularizer=None,
                                  name="decoder_bi_gru2_layer")
    decoder_gru2 = decoder_gru2_layer(decoder_gru1)

    decoder_gru3_layer = GRU(latent_dim * 2,
                                  stateful=False,
                                  return_sequences=True,
                                  kernel_constraint=None,
                                  kernel_regularizer=None,
                                  name="decoder_bi_gru3_layer")
    decoder_outputs = decoder_gru3_layer(decoder_gru2)

    return decoder_outputs


def encoder_bi_GRU_gpu(input_shape, encoder_inputs, latent_dim):
    encoder = Bidirectional(CuDNNGRU(latent_dim,
                                     input_shape=(None, input_shape),
                                     stateful=False,
                                     return_sequences=False,
                                     return_state=True,
                                     kernel_constraint=None,
                                     kernel_regularizer=None,
                                     recurrent_initializer='glorot_uniform'),
                            name="encoder_bi_gru_layer")
    # 'encoder_outputs' are ignored and only states are kept.
    encoder_outputs, forward_h, backward_h = encoder(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    encoder_states = [state_h]

    return encoder_states


def decoder_for_bidirectional_encoder_GRU_gpu(target_length, encoder_states, decoder_inputs, latent_dim):
    # First Layer
    decoder_gru1_layer = CuDNNGRU(latent_dim * 2,
                                   input_shape=(None, target_length),
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_constraint=None,
                                   kernel_regularizer=None,
                                   name="decoder_bi_gru1_layer")
    decoder_gru1,state_h = decoder_gru1_layer(decoder_inputs, initial_state=encoder_states)

    # Second LSTM Layer
    decoder_gru2_layer = CuDNNGRU(latent_dim * 2,
                                   stateful=False,
                                   return_sequences=True,
                                   kernel_constraint=None,
                                   kernel_regularizer=None,
                                   name="decoder_bi_gru2_layer")
    decoder_gru2 = decoder_gru2_layer(decoder_gru1)

    decoder_gru3_layer = CuDNNGRU(latent_dim * 2,
                                   stateful=False,
                                   return_sequences=True,
                                   kernel_constraint=None,
                                   kernel_regularizer=None,
                                   name="decoder_bi_gru3_layer")
    decoder_outputs = decoder_gru3_layer(decoder_gru2)

    return decoder_outputs

