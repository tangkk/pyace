# this is a song transposition program
# run it such as "python songtranspose.py infile C outfile E"
# which will transpose the infile from C key to E key
# run "python songtranspose.py infile C outfile allinone" will place all different keys transposition in one file
# run "python songtranspose.py infile C outfile all" will place all different keys transposition in different files
# written by tangkk

import sys
import io

def songtranspose(insong, oldkey, outsong, newkey):
    diatonic = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    key_sharp = ['C', 'C#', 'D', 'D#', 'E','F#', 'G', 'G#', 'A', 'A#', 'B']
    key_flat = ['Db','Eb','F','Gb','Ab', 'Bb']
    chromatic_sharp = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chromatic_flat = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    chromatic_sharp_m = ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm']
    chromatic_flat_m = ['Cm', 'Dbm', 'Dm', 'Ebm', 'Em', 'Fm', 'Gbm', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm']

    if 'm' in oldkey:
        if oldkey in chromatic_sharp_m:
            oldkeyIdx = chromatic_sharp_m.index(oldkey)
        else:
            oldkeyIdx = chromatic_flat_m.index(oldkey)
    else:
        if oldkey in chromatic_sharp:
            oldkeyIdx = chromatic_sharp.index(oldkey)
        else:
            oldkeyIdx = chromatic_flat.index(oldkey)


    def transposechord(inchord, transpose, newkey):
        suffix = ''
        inroot = ''
        outroot = ''
        outchord = ''
        inIdx = 0

        # stripe out the suffix of a chord
        if 'b' in inchord:
            inroot = inchord[0] + inchord[1]
            if len(inchord) > 2:
                suffix = inchord[2:len(inchord)]
        elif '#' in inchord:
            inroot = inchord[0] + inchord[1]
            if len(inchord) > 2:
                suffix = inchord[2:len(inchord)]
        else:
            inroot = inchord[0]
            if len(inchord) > 1:
                suffix = inchord[1:len(inchord)]

        # transform equivalent root symbols
        if inroot == 'E#':
            inroot = 'F'
        if inroot == 'B#':
            inroot = 'C'
        if inroot == 'Fb':
            inroot = 'E'
        if inroot == 'Cb':
            inroot = 'B'

        if inroot in chromatic_sharp:
            inIdx = chromatic_sharp.index(inroot)
        else:
            inIdx = chromatic_flat.index(inroot)

        outIdx = (inIdx + transpose) % 12

        if newkey in key_flat:
            outroot = chromatic_flat[outIdx]
        else:
            outroot = chromatic_sharp[outIdx]

        outchord = outroot + suffix

        return outchord

    def transposenewkey(newkey, inchord):
        if 'm' in newkey:
            if newkey in chromatic_sharp_m:
                newkeyIdx = chromatic_sharp_m.index(newkey)
            else:
                newkeyIdx = chromatic_flat_m.index(newkey)
        else:
            if newkey in chromatic_sharp:
                newkeyIdx = chromatic_sharp.index(newkey)
            else:
                newkeyIdx = chromatic_flat.index(newkey)

        transpose = newkeyIdx - oldkeyIdx

        if '/' in inchord:
            inchord = inchord.split('/')
            outchord = transposechord(inchord[0], transpose, newkey) + '/' + transposechord(inchord[1], transpose, newkey)
        else:
            outchord = transposechord(inchord, transpose, newkey)

        return outchord

    # transpose the song
    def transposesong(newkey, insong):
        newsong = ''
        newline = ''
        newtoken = ''
        f = io.open(insong, encoding="utf8")
        for line in f:
            idx = 0
            record = 0
            newline = ''
            newtoken = ''
            oldtoken = ''
            forbid = 0
            record = 0
            for token in line:
                forbid = 0
                newtoken = token

                if record == 1 and (token == '#' or token == 'b'):
                    newtoken = oldtoken+token
                    newtoken = transposenewkey(newkey, newtoken)
                    record = 0
                    forbid = 0
                elif record == 1 and token != '#' and token != 'b':
                    newtoken = transposenewkey(newkey, oldtoken)
                    newtoken = newtoken+token
                    record = 0
                    forbid = 0

                if token in diatonic:
                    oldtoken = newtoken
                    record = 1
                    forbid = 1

                if forbid == 0:
                    newline+=newtoken

                oldtoken = newtoken

            newsong+=newline
        f.close()
        return newsong


    if (newkey in chromatic_sharp) or (newkey in chromatic_flat):
        newsong = transposesong(newkey, insong)
        fw = io.open(outsong,'w', encoding="utf8")
        fw.write(newsong)
        fw.close()

    if newkey == 'all':
        for key in chromatic_sharp:
            newsong = transposesong(key, insong)
            fw = io.open(outsong+'_'+key,'w', encoding="utf8")
            fw.write(newsong)
            fw.close()
        for key in chromatic_flat:
            newsong = transposesong(key, insong)
            fw = io.open(outsong+'_'+key,'w', encoding="utf8")
            fw.write(newsong)
            fw.close()

    if newkey == 'allinone':
        allnewsongs= ''
        for key in chromatic_sharp:
            newsong = transposesong(key, insong)
            allnewsongs += newsong
        for key in chromatic_flat:
            newsong = transposesong(key, insong)
            allnewsongs += newsong
        fw = io.open(outsong,'w', encoding="utf8")
        fw.write(allnewsongs)
        fw.close()
        
if __name__ == "__main__":
    insong = sys.argv[1]
    oldkey = sys.argv[2]
    outsong = sys.argv[3]
    newkey = sys.argv[4]
    songtranspose(insong, oldkey, outsong, newkey)