import xml.etree.ElementTree as ET

from lib import Transcript


def get_transcriptions(file_path):
    """
    Gets the list of transcriptions that corresponds to one audio fine
    :param file_path: String
    :return:
    """
    transcriptions = []

    tree = ET.parse(file_path)
    root = tree.getroot()
    for body in root.find("body"):

        for segment in body.findall("segment"):
            start_time = float(segment.attrib["starttime"])
            end_time = float(segment.attrib["endtime"])
            transcript = ""
            for element in segment.findall("element"):
                elt = element.text
                if elt is not None:
                    transcript += elt + " "
            transcriptions.append(Transcript(start_time, end_time, transcript))
    return transcriptions


def generate_transcriptions_file(transcriptions_desc, output_path):
    """
    Creates transcription files
    :param transcriptions_desc: String
    :param output_path: String
    :return:
    """
    transcriptions_text = []
    for t in transcriptions_desc:
        transcriptions_text.append(t.content)

    with open(output_path, "w") as output_file:
        output_file.write(transcriptions_text[0])
        transcriptions_text.pop(0)
        for line in transcriptions_text:
            output_file.write("\n" + line)
        output_file.close()
