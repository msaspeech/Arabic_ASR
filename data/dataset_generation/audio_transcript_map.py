import re
from etc import settings
from utils import get_files_full_path, read_file_content


def get_transcript_key_value_test_dataset(transcript):
    """
    Retreives file name and its transcript for each line of the transcriptions file
    :param transcript: String
    :return: String, String
    """
    transcript = transcript.rstrip("\n")
    key_pattern = "(MSA[\d]+) .*"
    value_pattern = "MSA[\d]+ (.*)"
    transcript_wav_file = re.findall(key_pattern, transcript)[0] + ".wav"
    transcript_content = re.findall(value_pattern, transcript)[0]

    return transcript_wav_file, transcript_content


def map_transcripts_test_dataset(file_path):
    """
    Generates a dict where the key is the audio file name and the value is its transcript
    :param file_path: String
    :return: Dict
    """
    transcripts_map = dict()
    with open(file_path, "r") as transcripts_file:
        uncleaned_transcripts = transcripts_file.readlines()
        for t in uncleaned_transcripts:
            transcript_key, transcript_value = get_transcript_key_value_test_dataset(t)
            transcripts_map[transcript_key] = transcript_value
    return transcripts_map


def map_audio_transcripts_xml():
    """
    Maps an audio folder to its transcription file
    :return: Dict
    """
    audio_transcripts_map = dict()

    transcript_file_paths = get_files_full_path(settings.GENERATED_DATA_TRANSCRIPTS_PATH)
    for i, transcript_file in enumerate(transcript_file_paths):
        transcriptions = read_file_content(transcript_file)
        for j, transcript in enumerate(transcriptions):
            transcript = transcript.rstrip()
            audio_file_path = settings.GENERATED_DATA_WAV_PATH + "audio" + str(i) + "/track" + str(j) + ".wav"
            audio_transcripts_map[audio_file_path] = transcript

    return audio_transcripts_map


def map_audio_transcripts_generic():
    """
    Maps an audio folder to its transcription file
    :return: Dict
    """
    audio_transcripts_map = dict()

    transcript_file_paths = get_files_full_path(settings.TRANSCRIPTIONS_DATA_PATH)
    audio_file_paths = get_files_full_path(settings.AUDIO_DATA_PATH)

    for i, transcript_file in enumerate(transcript_file_paths):
        transcriptions = read_file_content(transcript_file)
        for j, transcript in enumerate(transcriptions):
            transcript = transcript.rstrip()
            audio_file_path = audio_file_paths[j]
            audio_transcripts_map[audio_file_path] = transcript

    return audio_transcripts_map