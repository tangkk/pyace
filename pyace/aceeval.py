# this script uses the MusOOEvaluator to evaluate the ACE's performance
# https://github.com/jpauwels/MusOOEvaluator

from subprocess import call
import sys

if __name__ == "__main__":
    testpath = sys.argv[1]
    fw = open('evallist','w')
    with open(testpath) as f:
        for line in f:
            line = line.rstrip("\n")
            line = line.rstrip("\r")
            if '.mp3' in line:
                line = line.rstrip(".mp3")
            if '.wav' in line:
                line = line.rstrip(".wav")
            fw.write(line+"\n")
    fw.close()

    resroot = '../data/res'
    labelroot = '../data/label'
    sep = '/'

    # ************
    # evaluation
    # ************
    call(['sh','eval.sh','evallist','ace'])

