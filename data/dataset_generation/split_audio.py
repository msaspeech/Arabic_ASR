from pydub import AudioSegment


def split_audio(audio_entry, transcriptions_desc, audio_output_dir):
    """
    Takes audio sample and its time informatins and splits one large audio sample to shorter audio samples
    :param audio_entry: String
    :param transcriptions_desc: List of Transcript
    :param audio_output_dir: String
    :return:
    """
    for i, transcript in enumerate(transcriptions_desc):
        start_time = transcript.start_time * 1000
        end_time = transcript.end_time * 1000
        audio = AudioSegment.from_wav(audio_entry)
        audio = audio[start_time:end_time]
        file_name = audio_output_dir + "track" + str(i) + ".wav"
        audio.export(file_name, format="wav")
