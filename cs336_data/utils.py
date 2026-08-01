import os
import re
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from fastwarc.stream_io import GZipStream, FileStream


def extract_text(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)
    if encoding is None:
        raise ValueError("Could not detect encoding")
    html_str = html_bytes.decode(encoding)
    return extract_plain_text(html_str)


def warc_to_text(warc_file, num_files = 1):
    stream = GZipStream(FileStream(warc_file, 'rb'))
    for i, record in enumerate(ArchiveIterator(stream, record_types=WarcRecordType.response)):
        html_bytes = record.reader.read()
        text = extract_text(html_bytes)
        print(f"Extracted text: {text}")
        if i + 1 == num_files:
            break


def identify_language(text):
    text = text.replace("\n", "")
    language_classifier = fasttext.load_model('shared-data/classifiers/lid.176.bin')
    prediction = language_classifier.predict(text)
    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    print(f"Language label: {label}, confidence: {confidence}")
    return label, confidence


def mask_emails(text):
    replace_with = "|||EMAIL_ADDRESS|||"
    regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    masked_text, count = re.subn(regex, replace_with, text, flags = re.IGNORECASE)
    return masked_text, count


def mask_phone_numbers(text):
    replace_with = "|||PHONE_NUMBER|||"
    regex = r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}"
    masked_text, count = re.subn(regex, replace_with, text, flags = re.IGNORECASE)
    return masked_text, count


def mask_ips(text):
    replace_with = "|||IP_ADDRESS|||"
    regex = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    masked_text, count = re.subn(regex, replace_with, text, flags = re.IGNORECASE)
    return masked_text, count



if __name__ == "__main__":
    # warc_to_text('shared-data/CC/example.warc.gz')

    # identify_language("hi my name is bob")

    masked_text, count = mask_emails("hi my name is bob and u can contact me at bob123@gmail.com or hi@yahoo.com")
    print(f"masked_text: {masked_text}, count: {count}")

    masked_text, count = mask_phone_numbers("hi my name is bob and u can contact me at 123-456-7890")
    print(f"masked_text: {masked_text}, count: {count}")

