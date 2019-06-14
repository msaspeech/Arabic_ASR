# from models import | Word_Inference_TEST
#
# word_level = 1
# architecture = 1
# latent_dim = 350
#
# from etc import settings
# from lib import AudioInput
# import numpy as np
#
# # settings.DRIVE_INSTANCE = load_pickle_data(DRIVE_INSTANCE_PATH)
#
#
# model_name = "architecture" + str(architecture)
# if word_level:
#     model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "word.h5"
# else:
#     model_path = settings.TRAINED_MODELS_PATH + model_name + "/" + model_name + "char.h5"
#
# sample = AudioInput("audio.wav", "")
# audio = [sample.mfcc.transpose()]
#
# audio_sequence = np.array(audio, dtype=np.float32)
# print(audio_sequence.shape)
#
# word_inference = Word_Inference_TEST(model_path=model_path, latent_dim=latent_dim)
# word_inference.predict_sequence_test(audio_input=audio_sequence)
# # 9word_inference.test_encoder_decoder(audio_sequence)
# word_inference.decode_audio_sequence(audio_sequence)
#
# # word_inference2 = Word_Inference(model_path=model_path, latent_dim=latent_dim)
# # transcript = word_inference.decode_audio_sequence(audio_sequence)
# # print(transcript)
#
# # accuracy = measure_test_accuracy(test_decoder_input, model, encoder_states, latent_dim=512)
