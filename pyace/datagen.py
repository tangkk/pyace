# segmentation then classification approach to ACE
# feature extraction: use librosa's cqt, chromagram (https://github.com/librosa/librosa)
# segmentation: use hmmlearn (http://hmmlearn.readthedocs.io/en/stable/), which has a sklean interface
# classification: use keras - theano/tensorflow backend (https://keras.io/)

import librosa
import librosa.display
import numpy
from numpy import inf
import cPickle
import sys
import os
import math
from hmmlearn import hmm
import matplotlib.pyplot as plt
import h5py

# customized modules
from ..fe import fe
from ..helpers import helpers

if __name__ == '__main__':
    tracklists = ['C','J','K','U','R','B']

    for tl in tracklists:
        Xcg = []
        Xcqt = []
        Ycg = []
        Ycqt = []
        tpath = '../data/tracklists/' + tl

        with open(tpath) as tf:
            for tline in tf:
                print tline
                tline = tline.rstrip("\n")
                tline = tline.rstrip("\r")
                tline = tline[:-4]
                audiopath = '../data/audio/' + tline + '.mp3'
                labelpath = '../data/label/' + tline + '.lab'

                # hyper params
                sr = 22050
                hopsize = 512
                framedur = 512 / 22050.0
                nchords = helpers.nchords
                nroots = helpers.nroots

                # *******************************************
                # feature extraction - use librosa
                # *******************************************
                print "feature extraction..."
                cqtspec, chromagram = fe.fe(audiopath, sr, hopsize)

                print "ground truth segmentation..."
                with open(os.path.realpath(labelpath)) as f:
                    for line in f:
                        line = line.rstrip("\n")
                        line = line.rstrip("\r")
                        tokens = line.split("\t")
                        if len(tokens) < 3:
                            tokens = line.split(" ")
                        st = float(tokens[0])
                        et = float(tokens[1])
                        oc = tokens[2]
                        print tokens

                        # majmin chord mapping
                        chord, isminor = helpers.mmchordmaping(oc)
                        # print 'chord',chord

                        # chord name to chord num
                        chordnum, rootnum = helpers.chordname2chordnum(chord)
                        # print 'chordnum',chordnum
                        # print 'rootnum',rootnum

                        # segment tiling (N=6) -> 6*n_feature
                        cgfeature = helpers.segtile(6, chromagram, st, et, framedur)
                        # print 'cgfeature', cgfeature

                        # data augmentation - 12 times
                        for i in range(12): # from 0 to 12
                            # round shift the feature
                            cgrfeature = numpy.roll(cgfeature,i,axis=1)
                            Xcg.append(cgrfeature)

                            # round shift the label
                            rrootnum = (rootnum + i) % 12
                            rchordnum = rrootnum + isminor*12
                            if chord is not 'N':
                                Ycg.append(rchordnum)
                                # print 'rchord', helpers.chordnames[rchordnum]
                            else:
                                Ycg.append(nchords)
                                # print 'rchord', 'N'

                            # print cgrfeature

                        # segment tiling (N=6) -> 6*n_feature
                        cqtfeature = helpers.segtile(6, cqtspec, st, et, framedur)

                        # data augmentation - 12 times
                        for i in range(12): # from -5 to 6 (times 3 bins)
                            # round shift the feature
                            ccqtfeature = numpy.roll(cqtfeature,3*(i-5),axis=1) # without zero padding
                            Xcqt.append(ccqtfeature)

                            # round shift the label
                            rrootnum = (rootnum + i-5) % 12
                            rchordnum = rrootnum + isminor * 12
                            if chord is not 'N':
                                Ycqt.append(rchordnum)
                            else:
                                Ycqt.append(nchords)

        h5f = h5py.File(tl+'.h5', 'w')
        h5f.create_dataset('Xcg', data=Xcg)
        h5f.create_dataset('Ycg', data=Ycg)
        h5f.create_dataset('Xcqt', data=Xcqt)
        h5f.create_dataset('Ycqt', data=Ycqt)
        h5f.close()
        del Xcg
        del Ycg
        del Xcqt
        del Ycqt

