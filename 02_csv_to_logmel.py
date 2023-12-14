import pandas as pd
import numpy as np
import librosa
from IPython.display import Audio
import matplotlib.pyplot as plt
# Essentia library: https://essentia.upf.edu/
from essentia import Pool
import essentia.standard as es
from tqdm.auto import tqdm

def log_mel_spectrogram(filename, sr=16000, n_mels=64):
    audio = es.MonoLoader(filename=filename, sampleRate=sr)()
    audio_noise = np.zeros(10*sr, dtype='float32')
    audio_noise[:audio.size] = audio
    audio_noise = audio_noise + 5*1e-4*np.random.rand(audio_noise.size).astype('float32')
    audio = audio_noise

    windowing = es.Windowing(type='hann', normalized=False, zeroPadding=0)
    spectrum = es.Spectrum()
    melbands = es.MelBands(numberBands=n_mels, 
                           sampleRate=sr, 
                           lowFrequencyBound=0, 
                           highFrequencyBound=8000, 
                           inputSize=(2048)//2+1, 
                           weighting='linear', 
                           normalize='unit_tri', 
                           warpingFormula='slaneyMel', 
                           type='power')
    
    norm10k = es.UnaryOperator(type='identity', shift=0, scale=1)
    log10 = es.UnaryOperator(type='log10')
    results = Pool()

    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=False):
        spectrumFrame = spectrum(windowing(frame))     
        results.add('melbands', log10(norm10k(melbands(spectrumFrame))))
    
    mel_log_spectrum = results['melbands']
    return mel_log_spectrum[:n_mels].T

csv_file = 'dataset.csv'  
df = pd.read_csv(csv_file)

# x = log_mel_spectrogram(f'../IRMAS-TrainingData/voi/[voi][jaz_blu]2334__1.wav')
# librosa.display.specshow(x, sr=16000, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Log Mel Spectrogram')
# plt.xlabel('Time')
# plt.ylabel('Mel Frequency')
# plt.savefig('log_mel_spectrogram.png')

num_sounds = len(df)
log_mel_spectrograms = np.zeros((num_sounds, 64, 64))
labels = np.zeros((num_sounds,))

with tqdm(total=num_sounds) as pbar:
    for idx, row in df.iterrows():
        filename = f"../IRMAS-TrainingData/{row['label']}/{row['filename']}"
        label = row['label']
        label_number = row['label_number']
        labels[idx] = label_number
        log_mel_spectrograms[idx] = log_mel_spectrogram(filename)
        pbar.update()

np.save('log_mel_spectrograms.npy', log_mel_spectrograms)
np.save('labels.npy', labels)