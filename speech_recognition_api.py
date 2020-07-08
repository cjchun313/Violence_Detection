#-*- coding:utf-8 -*-
import soundfile as sf
import numpy as np
import speech_recognition as sr
import librosa

import urllib3
import json
import base64

import requests

class ClovaSR:
    def __init__(self):
        self.client_id = ""
        self.client_secret = ""
        self.lang = "Kor"  # 언어 코드 ( Kor, Jpn, Eng, Chn )
        self.url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + self.lang

        self.headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
            "Content-Type": "application/octet-stream"
        }

        self.target_fs = 16000
        self.tmp_path = '../db/tmp.wav'

    def read_audio(self, filepath):
        (audio, fs) = sf.read(filepath)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if fs != self.target_fs:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.target_fs)
            fs = self.target_fs

        sf.write(self.tmp_path, audio, samplerate=fs, subtype='PCM_16')
        audio = open(self.tmp_path, 'rb')

        return audio

    def transcript_audio(self, audio):
        response = requests.post(self.url, data=audio, headers=self.headers)

        if response.status_code != 200:
            print("Error : " + response.text)

        return self.crop_only_characters(response.text)

    def crop_only_characters(self, data):
        a = data.find('{"text":"')
        b = data.find('"}')

        return data[a+9:b-0]



class GoogleWebSR:
    def __init__(self):
        self.languageCode = 'ko-KR'
        self.target_fs = 16000
        self.tmp_path = '../db/tmp.wav'
        self.r = sr.Recognizer()

    def read_audio(self, filepath):
        (audio, fs) = sf.read(filepath)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if fs != self.target_fs:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.target_fs)
            fs = self.target_fs

        sf.write(self.tmp_path, audio, samplerate=fs, subtype='PCM_16')
        audio = sr.AudioFile(self.tmp_path)

        with audio as source:
            sig = self.r.record(source)

        return sig

    def transcript_audio(self, audio):
        ASR_result = self.r.recognize_google(audio, language=self.languageCode, show_all=True)
        # ASR_result = self.r.recognize_google(audio,language="ko-KR")
        ASR_result_text = ASR_result['alternative'][0]['transcript']
        #ASR_result_confidence = ASR_result['alternative'][0]['confidence']

        #return ASR_result_text, ASR_result_confidence
        return ASR_result_text


class EtriSR:
    def __init__(self):
        self.openApiURL = 'http://aiopen.etri.re.kr:8000/WiseASR/Recognition'
        self.accessKey = ''
        self.languageCode = 'korean'
        self.target_fs = 16000
        self.tmp_path = '../db/tmp.wav'

    def read_audio(self, filepath):
        (audio, fs) = sf.read(filepath)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if fs != self.target_fs:
            audio = librosa.resample(audio, orig_sr=fs, target_sr=self.target_fs)
            fs = self.target_fs

        sf.write(self.tmp_path, audio, samplerate=fs, subtype='PCM_16')
        (audio, fs) = sf.read(self.tmp_path, dtype='int16')

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

    def transcript_audio(self, audio):
        requestJson = self.generate_requestJson(audio)
        response = self.decode_audio(requestJson)

        if response.status != -1:
            data = self.crop_only_characters(str(response.data, 'utf-8'))
            #print(data)
        else:
            print(response.status)
            print('error to decode the audio for speech recognition!')

        return data

if __name__ == "__main__":
    filepath = '../db/test/sample1_short2.wav'

    etri = EtriSR()
    audio = etri.read_audio(filepath)
    txt = etri.transcript_audio(audio)
    print('etri:', txt)

    gw = GoogleWebSR()
    audio = gw.read_audio(filepath)
    txt = gw.transcript_audio(audio)
    print('google web:', txt)

    csr = ClovaSR()
    audio = csr.read_audio(filepath)
    txt = csr.transcript_audio(audio)
    print('clova:', txt)





