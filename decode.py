from models.Speech_API.models.speech_recognition import recognize_speech

decoded_sentence = recognize_speech("test.wav", latent_dim=300, architecture=1)
print(decoded_sentence)
