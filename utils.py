import numpy as np
import librosa
import soundfile as sf

def read_audio(filepath, target_fs=16000, tmp_path='../db/tmp.wav'):
    (audio, fs) = sf.read(filepath)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    sf.write(tmp_path, audio, samplerate=fs, subtype='PCM_16')
    (audio, fs) = sf.read(tmp_path, dtype='int16')

    return audio

