#-*- coding:utf-8 -*-
import soundfile
import numpy as np

import urllib3
import json
import base64

class ETRISpeechRecognition:
    def __init__(self):
        self.openApiURL = 'http://aiopen.etri.re.kr:8000/WiseASR/Recognition'
        self.accessKey = '8a72eaa1-f00d-4339-b2d0-9a05004b383a'
        self.languageCode = 'korean'

    def read_audio(self, filepath):
        (audio, fs) = soundfile.read(filepath, dtype='int16')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if fs != 16000:
            print('sampling frequency must be 16 kHz!')
            return -1

        audio = base64.b64encode(audio).decode("utf8")

        return audio

    def generate_requestJson(self, audio):
        requestJson = {
            "access_key": self.accessKey,
            "argument": {
                "language_code": self.languageCode,
                "audio": audio
            }
        }

        return requestJson

    def decode_audio(self, requestJson):
        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            self.openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        )

        return response

    def crop_only_characters(self, data):
        a = data.find('"recognized":')
        b = data.find('}}')

        return data[a+14:b-1]

if __name__ == "__main__":
    etri_sr = ETRISpeechRecognition()

    filepath = '../db/test/KsponSpeech_000001_01.wav'
    audio = etri_sr.read_audio(filepath)
    requestJson = etri_sr.generate_requestJson(audio)
    response = etri_sr.decode_audio(requestJson)

    if response.status != -1:
        data = etri_sr.crop_only_characters(str(response.data, 'utf-8'))
        print(data)
    else:
        print(response.status)
        print('error to decode the audio for speech recognition!')


