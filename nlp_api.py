# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

from speech_recognition_api import GoogleCloudSR

class GoogleCloudNLP:
    def __init__(self):
        # Instantiates a client
        self.client = language.LanguageServiceClient()

    def analyzeSeq(self, seq):
        # The text to analyze
        document = types.Document(
            content=seq,
            language='ko',
            type=enums.Document.Type.PLAIN_TEXT)

        # Detects the sentiment of the text
        sentiment = self.client.analyze_sentiment(document=document).document_sentiment

        return sentiment

if __name__ == "__main__":
    filepath = '../db/test/file003_e.wav'

    gc = GoogleCloudSR()
    audio = gc.read_audio(filepath)
    response = gc.transcript_audio(audio)

    gn = GoogleCloudNLP()

    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
        sentiment = gn.analyzeSeq(alternative.transcript)
        print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))

        # Print the start and end time of each word
        for i, word in enumerate(alternative.words):
            if i == 0:
                print(u"Start time: {} seconds {} nanos".format(word.start_time.seconds, word.start_time.nanos))

            if i == (len(alternative.words) - 1):
                print(u"End time: {} seconds {} nanos".format(word.end_time.seconds, word.end_time.nanos))

    print('done!')
