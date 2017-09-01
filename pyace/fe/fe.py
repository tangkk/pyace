# this is for feature extraction

import librosa
import numpy
from numpy import inf

def fe(audiopath, sr, hopsize):
    # load audio
    y, _ = librosa.core.load(audiopath, sr=sr, mono=True)

    # constant-Q spectrogram
    cqtspec = librosa.core.cqt(y=y, sr=sr, hop_length=hopsize, fmin=None,
                               n_bins=252, bins_per_octave=36, tuning=0.0, filter_scale=1,
                               norm=1, sparsity=0.01, window='hann', scale=True)
    cqtspec = numpy.transpose(cqtspec)  # transpose for interfacing with the hmm: (n_samples,n_features)

    # chromagram
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, C=None,
                                            hop_length=hopsize, fmin=None, norm=inf, threshold=0.0,
                                            tuning=None, n_chroma=12, n_octaves=7, window=None,
                                            bins_per_octave=None, cqt_mode='full')
    chromagram = numpy.transpose(chromagram)  # transpose for interfacing with the hmm: (n_samples,n_features)

    return cqtspec, chromagram