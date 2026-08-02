import os
import re
import fasttext
import nltk.tokenize
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


def detect_nsfw(text):
    text = text.replace("\n", "")
    nsfw_classifier = fasttext.load_model('shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin')
    prediction = nsfw_classifier.predict(text)
    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    print(f"Language label: {label}, confidence: {confidence}")
    return label, confidence


def detect_toxic_speech(text):
    text = text.replace("\n", "")
    toxic_speech_classifier = fasttext.load_model('shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin')
    prediction = toxic_speech_classifier.predict(text)
    label = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    print(f"Language label: {label}, confidence: {confidence}")
    return label, confidence


def gopher_filter(text):
    text = text.strip()
    tokenized = nltk.tokenize.word_tokenize(text)
    tokenized = [token for token in tokenized if token.strip()]

    if len(tokenized) < 50 or len(tokenized) > 100000:
        print("Contains less than 50 or more than 100,000 words")
        return False

    mean_length = sum(len(word) for word in tokenized) / len(tokenized)
    if mean_length < 3 or mean_length > 10:
        print("Mean word length outside the range of 3 to 10 characters")
        return False

    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    if (sum(1 for line in lines if line.endswith("...")) / len(lines)) > 0.3:
        print("Has more than 30% of lines ending with an ellipsis")
        return False

    alpha_char_count = sum(1 for word in tokenized if any(c.isalpha() for c in word))
    alpha_frac = alpha_char_count / len(tokenized)
    if alpha_frac < 0.8:
        print("Contain less than 80% of words with at least one alphabetic character")
        return False

    print("All filters passed!")
    return True



if __name__ == "__main__":
    # warc_to_text('shared-data/CC/example.warc.gz')

    # identify_language("hi my name is bob")

    # masked_text, count = mask_emails("hi my name is bob and u can contact me at bob123@gmail.com or hi@yahoo.com")
    # print(f"masked_text: {masked_text}, count: {count}")

    # masked_text, count = mask_phone_numbers("hi my name is bob and u can contact me at 123-456-7890")
    # print(f"masked_text: {masked_text}, count: {count}")

    gopher_filter("hi my name is bob")
