# a bunch of helper functions

import numpy
import math

nchords = 24
nroots = 12

chordlabel2num = {
    'C':0,'B#':0,
	'C#':1,'Db':1,
	'D':2,
	'D#':3,'Eb':3,
    'E':4,'Fb':4,
    'F':5,'E#':5,
	'F#':6,'Gb':6,
	'G':7,
	'G#':8,'Ab':8,
	'A':9,
	'A#':10,'Bb':10,
    'B':11,'Cb':11,
    'Cm':12,'B#m':12,
	'C#m':13,'Dbm':13,
	'Dm':14,
	'D#m':15,'Ebm':15,
    'Em':16,'Fbm':16,
    'Fm':17,'E#m':17,
	'F#m':18,'Gbm':18,
	'Gm':19,
	'G#m':20,'Abm':20,
	'Am':21,
	'A#m':22,'Bbm':22,
    'Bm':23,'Cbm':23
	}

rootlabel2num = {
    'C':0,'B#':0,
	'C#':1,'Db':1,
	'D':2,
	'D#':3,'Eb':3,
    'E':4,'Fb':4,
    'F':5,'E#':5,
	'F#':6,'Gb':6,
	'G':7,
	'G#':8,'Ab':8,
	'A':9,
	'A#':10,'Bb':10,
    'B':11,'Cb':11
}

chordnames = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B',
              'Cm','C#m','Dm','D#m','Em','Fm','F#m','Gm','G#m','Am','A#m','Bm','N']

def mmchordmaping(oc):
	ctokens = oc.split('/')
	oc = ctokens[0]
	ctokens = oc.split(':')
	root = ctokens[0]
	quality = []
	isminor = 0
	if len(ctokens) == 2:
		quality = ctokens[1]
	if 'm' in quality and 'maj' not in quality:
		chord = root + 'm'
		isminor = 1
	else:
		chord = root

	return chord, isminor

def chordname2chordnum(chord):
	chordnum = 0
	rootnum = 0
	if chord is not 'N':
		chordnum = chordlabel2num[chord]
		rootnum = chordnum % 12
	else:
		chordnum = nchords
		rootnum = nroots

	return chordnum, rootnum

def segtile(numtile, ingram, st, et, framedur):
	sf = long(st / framedur)  # as number of frames
	ef = long(et / framedur)  # as number of frames
	feature = numpy.zeros((numtile, ingram.shape[1]))
	buf = ingram[sf:ef]
	lenseg = ef - sf + 1
	sixseg = int(math.ceil(lenseg / 6))

	for i in range(6):
		seg = buf[i * sixseg:min((i + 1) * sixseg, lenseg)]
		segfeature = numpy.mean(seg, axis=0)
		feature[i] = segfeature

	return feature
