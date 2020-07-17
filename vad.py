import webrtcvad
import numpy as np
import matplotlib.pyplot as plt

from utils import read_audio

class WebrtcVAD:
    def __init__(self, mode=3):
        self.sample_rate = 16000
        #self.window_duration = 0.020  # duration in seconds

        self.samples_per_window = 640
        # samples_per_window = int(window_duration * sample_rate + 0.5)
        #self.bytes_per_sample = 2

        self.vad = webrtcvad.Vad()

        # set aggressiveness from 0 to 3
        self.vad.set_mode(mode)

    def perform_vad(self, audio):
        vad_res = []
        for start in np.arange(0, len(audio), self.samples_per_window):
            stop = min(start + self.samples_per_window, len(audio))
            #print(start, stop, len(audio))
            if (stop - start) < self.samples_per_window:
                break
            else:
                is_speech = self.vad.is_speech(audio[start:stop], sample_rate=self.sample_rate)
                if is_speech == True:
                    vad_res.append(1)
                else:
                    vad_res.append(0)

                #print(is_speech)

        return vad_res

if __name__ == "__main__":
    filepath = '../db/test/file003_e.wav'

    vad = WebrtcVAD()

    audio = read_audio(filepath)
    vad_res = vad.perform_vad(audio)

    vad_res = np.array(vad_res)
    print(vad_res)

    x = len(vad_res) * 640 / 16000
    x = np.linspace(0, x, len(vad_res))
    print(x.shape, vad_res.shape)

    plt.plot(x, vad_res)
    plt.xticks(np.arange(0, 42, step=2))
    plt.show()










