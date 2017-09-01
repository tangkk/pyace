# segmentation then classification approach to ACE
# feature extraction: use librosa's cqt, chromagram (https://github.com/librosa/librosa)
# segmentation: use hmmlearn (http://hmmlearn.readthedocs.io/en/stable/), which has a sklean interface
# classification: use keras - theano/tensorflow backend (https://keras.io/)

import librosa
import numpy
from numpy import inf
import cPickle
import sys
import os
from hmmlearn import hmm
from helpers import helpers
import keras
from keras.models import Sequential
from keras.models import Model
from chordlyrics import chordlyrics
from songtranspose import songtranspose

def hmmsetup():
    nchords = helpers.nchords # n_components
    # nchords = helpers.nchords + 1  # n_components (for nochords)
    nchroma = 12  # n_features
    model = hmm.GaussianHMM(n_components=nchords, covariance_type="diag")
    prior = numpy.ones(nchords) / numpy.linalg.norm(numpy.ones(nchords), 1)
    model.startprob_ = prior

    selftrans = 1e12
    trans = numpy.ones((nchords, nchords))
    for i in range(nchords):
        trans[i, i] = selftrans
    trans = trans / ((nchords - 1) + selftrans)
    model.transmat_ = trans

    maj = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]  # Cmaj
    min = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]  # Cmin
    # nochord = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # nochord (because the chromagram has been inf normed)

    # from Cmaj to Bmaj, then from Cmin to Bmin
    means = numpy.zeros((nchords, nchroma))
    for i in range(nchroma):  # for maj
        means[i] = numpy.roll(maj, i)
    for i in range(nchroma):  # for min
        means[i + nchroma] = numpy.roll(min, i)
    # means[-1] = nochord # added for nochord
    model.means_ = means

    covars = numpy.ones((nchords, nchroma)) * 0.2
    model.covars_ = covars

    return model

def cqtextraction(songpath, sr, hopsize):
    # load audio
    y, _ = librosa.core.load(songpath, sr=sr, mono=True)

    # constant-Q spectrogram
    cqtspec = librosa.core.cqt(y=y, sr=sr, hop_length=hopsize, fmin=None,
            n_bins=84, bins_per_octave=12, tuning=0.0, filter_scale=1,
            norm=1, sparsity=0.01, window='hann', scale=True)
    cqtspec = numpy.transpose(cqtspec) # transpose for interfacing with the hmm

    return cqtspec

def chromagramextraction(songpath, sr, hopsize):
    # load audio
    y, _ = librosa.core.load(songpath, sr=sr, mono=True)

    # chromagram
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr, C=None,
                                            hop_length=hopsize, fmin=None, norm=inf, threshold=0.0,
                                            tuning=None, n_chroma=12, n_octaves=7, window=None,
                                            bins_per_octave=None, cqt_mode='full')
    chromagram = numpy.transpose(chromagram)  # transpose for interfacing with the hmm

    return chromagram

def segmentation(chords, framedur):
    ochord = helpers.chordnames[chords[0]]
    st = 0
    et = 0
    segments = []
    for i in range(len(chords)):
        et = round(i * framedur, 3)
        nchord = helpers.chordnames[chords[i]]
        if (nchord != ochord):
            if 'm' in ochord:  # mirex convention
                ochord = ochord.replace('m', ':min')
            print st, et, ochord
            segments.append((st,et))
            st = et
        ochord = nchord

    return segments

def writeres(chords, framedur, respath):
    ochord = helpers.chordnames[chords[0]]
    st = 0
    et = 0
    fw = open(respath,'w')
    for i in range(len(chords)):
        et = round(i * framedur,3)
        nchord = helpers.chordnames[chords[i]]
        if (nchord != ochord):
            if 'm' in ochord: # mirex convention
                ochord = ochord.replace('m',':min')
            print st, et, ochord
            fw.write(str(st) + " " + str(et) + " " + ochord + "\n")
            st = et
        ochord = nchord
    fw.close()

def writesegres(Y, segments, respath):
    fw = open(respath,'w')
    for i in range(len(segments)):
        st = segments[i][0]
        et = segments[i][1]
        y = numpy.argmax(Y[i])
        ochord = 'N'
        if y==helpers.nchords:
            ochord = 'N'
        else:
            ochord = helpers.chordnames[y]
        if 'm' in ochord:  # mirex convention
            ochord = ochord.replace('m', ':min')
        print st, et, ochord
        fw.write(str(st) + " " + str(et) + " " + ochord + "\n")
    fw.close()

def simpleace(songpath, respath):
    # hyper params
    sr = 22050
    hopsize = 512
    framedur = 512 / 22050.0

    # *******************************************
    # feature extraction - use librosa
    # *******************************************
    print "feature extraction..."
    chromagram = chromagramextraction(songpath, sr, hopsize)

    # ****************************************************
    # segmentation - use an hmmlearn's GaussianHMM
    # ****************************************************
    print "hmm setup..."
    model = hmmsetup()

    print "hmm decoding..."
    chords = model.predict(chromagram)

    print "generating output..."
    writeres(chords, framedur, respath)

def deepace(songpath, respath, acemode, modelpath):
    # hyper params
    sr = 22050
    hopsize = 512
    framedur = 512 / 22050.0

    # *******************************************
    # feature extraction - use librosa
    # *******************************************
    print "feature extraction..."
    chromagram = chromagramextraction(songpath, sr, hopsize)

    # ****************************************************
    # segmentation - use an hmmlearn's GaussianHMM
    # ****************************************************
    print "hmm setup..."
    model = hmmsetup()

    print "generating segmentation..."
    chords = model.predict(chromagram)

    segments = segmentation(chords, framedur)

    # ****************************************************
    # classification - use the trained keras model
    # ****************************************************
    print "segment tiling..."
    X = []
    for seg in segments:
        st = seg[0]
        et = seg[1]
        cgfeature = helpers.segtile(6, chromagram, st, et, framedur)
        X.append(cgfeature)

    X = numpy.asarray(X)
    nsamples = X.shape[0]
    nseg = X.shape[1]
    nbin = X.shape[2]

    if acemode == 'fcnn':
        X = X.reshape(nsamples,nseg*nbin)

    print "classifying..."
    model = keras.models.load_model(modelpath)
    Y = model.predict(X)

    print "generating output..."
    writesegres(Y, segments, respath)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "please provide at least two arguments"
        print "try calling the program like:"
        print "python pyace.py [test_audio_path] [acemode] [modelpath]"
        quit()
    testpath = sys.argv[1] # the test tracklist or the test song
    acemode = sys.argv[2] # provide a model path or just use "simple"

    lyricpath = ''
    if acemode != 'simple':
        modelpath = sys.argv[3]
        if len(sys.argv) > 4:
            lyricpath = sys.argv[4]
    else:
        modelpath = ''
        if len(sys.argv) > 3:
            lyricpath = sys.argv[3]

    if '.mp3' in testpath or '.wav' in testpath or '.flac' in testpath:
        print "this is the single mode"
        if acemode == 'simple':
            simpleace(testpath,testpath+'.chords.txt')
        elif acemode == 'fcnn' or acemode == 'rnn':
            deepace(testpath, testpath+'.chords.txt', acemode, modelpath)
        else:
            print "ace model not supported!"
            quit()

        if len(lyricpath) > 0:
            print "combining chords and lyrics..."
            chordlyrics.chordlyrics(lyricpath, testpath+'.chords.txt', testpath+'.chordlyrics.txt')
    else: # this is the batch mode, make sure you have the audio and label contents put in ./data/
        print "this is the batch mode, make sure you have the audio and label contents put in ./data/ as required"
        resroot = '../data/res'
        songroot = '../data/audio'
        labelroot = '../data/label'
        sep = '/'

        if not os.path.exists(resroot):
            os.mkdir(resroot)

        # ************
        # prediction
        # ************
        with open(testpath) as f:
            for line in f:
                line = line.rstrip("\n")
                line = line.rstrip("\r")
                line = line.rstrip(".mp3") # only mp3 is allowed

                print "now processing..." + line
                tokens = line.split("/")
                artist = tokens[0]
                album = tokens[1]
                song = tokens[2]
                if not os.path.exists(resroot + sep + artist):
                    os.mkdir(resroot + sep + artist)
                if not os.path.exists(resroot + sep + artist + sep + album):
                    os.mkdir(resroot + sep + artist + sep + album)

                songpath = songroot + sep + line + ".mp3"
                respath = resroot + sep + line + ".txt"
                labelpath = labelroot + sep + line + ".lab"

                if acemode == 'simple':
                    simpleace(songpath, respath)
                elif acemode == 'rnn' or 'fcnn':
                    deepace(songpath, respath, acemode, modelpath)
                else:
                    print "ace model not supported!"
