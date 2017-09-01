# this is a program to combine automatic chord transcription and lyrics
# run it such as "python chordlyrics.py lyricsfile chordannotationfile outfile"
# for example run:
# python chordlyrics.py aihenjiandan.lrc aihenjiandan.txt aihenjiandan.ch
# written by tangkk

import sys
import io

def chordlyrics(inlyrics, inchords, outsheet):

    def get_sec(s):
        l1,l2 = s.split(':')

        if l2.find('.')==1:
            l21,l22 = l2.split('.')
        else:
            l21 = l2
            l22 = 0
        ret = float(l1) * 60 + float(l21) * 1 + float(l22)*0.6 / 100
        return ret

    def readlyrics(inlyrics):
        title = ''
        seclist = []
        lylist = []
        f = io.open(inlyrics, encoding="utf8")
        for line in f:
            sps = line.split(']')

            if len(sps)==2 and sps[0].find('ti') == 1:
                _,title = sps[0].split(':')
            elif len(sps)>=2 and (sps[0].find('ar')==1 or sps[0].find('al')==1
                                  or sps[0].find('by')==1 or sps[0].find('la')==1
                                  or sps[0].find('ve')==1 or sps[0].find('re')==1):
                continue
            elif len(sps)>=2:
                # append every sec and lyrics
                ly = sps[-1]
                for i in range(len(sps)-1):
                    sp1 = sps[i]
                    sec = get_sec(sp1[1:])
                    seclist.append(sec)
                    lylist.append(ly)
        f.close()
        # sort seclist and lylist in terms of seclist
        argsortidx = sorted(range(len(seclist)), key=lambda k: seclist[k])
        newseclist = sorted(seclist)
        newlylist = [lylist[i] for i in argsortidx]

        return title, newseclist, newlylist

    # note that the annotation does not fill the end time (only start time and the chord, separated by \t)
    # TODO: 1. modify the chords so that it match human's reading behavior
    def readchords(inchords):
        seclist = []
        chlist = []
        f = io.open(inchords, encoding="utf8")
        for line in f:
            tokens = line.split('	')
            if len(tokens) < 3:
                tokens = line.split(' ')
            seclist.append(float(tokens[0]))
            chlist.append(tokens[2])

        f.close()
        return seclist, chlist

    # main script
    title, lyseclist, lylist = readlyrics(inlyrics)
    chseclist, chlist = readchords(inchords)

    chcount = 0
    if len(title) == 0:
        outstr = ''
    else:
        outstr = title + '\n'

    # intro part
    curlysec = lyseclist[0]
    curline = ''
    while chcount < len(chseclist):
        j = chcount
        curchsec = chseclist[j]
        curch = chlist[j]
        if curchsec < curlysec:
            chcount += 1
            curline += curch.strip()
            curline += ' | '
        else:
            break
    curline += '\n'
    outstr += curline

    # the chord of this line of lyrics appears before the next line of lyrics
    for i in range(len(lyseclist)-1):
        curlysec = lyseclist[i]
        nextlysec = lyseclist[i+1]
        curly = lylist[i]
        nextly = lylist[i+1]
        curline = ''
        while chcount < len(chseclist):
            j = chcount
            curchsec = chseclist[j]
            curch = chlist[j]
            if curchsec >= curlysec and curchsec < nextlysec:
                chcount += 1
                curline += curch.strip()
                curline += ' | '
            else:
                break
        curline += '\n'
        curline += curly
        outstr += curline

    # outro
    curline = '\n'
    while chcount < len(chseclist):
        j = chcount
        curchsec = chseclist[j]
        curch = chlist[j]
        chcount += 1
        curline += curch.strip()
        curline += ' | '

    outstr += curline

    fw = io.open(outsheet,'w', encoding="utf8")
    fw.write(outstr)
    fw.close()
    
if __name__ == "__main__":
    inlyrics = sys.argv[1]
    inchords = sys.argv[2]
    outsheet = sys.argv[3]

    chordlyrics(inlyrics, inchords, outsheet)
    
    
    
    
    
    