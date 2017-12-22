from os import path as op
import re


# Notes from Mathieu Cliche https://arxiv.org/pdf/1704.06125.pdf,
# Winner of the SemEval 2017:
"""
    - URLS replaced by <url>
    - Emojis replaced by e.g <smile>, <sadface>, <lolface>, <neutralface>
    - Any letter repeated more than 2 times in a row, replaced by 2 repetitions. E.g "cooool" -> "cool"
    - All tweets lowercased
"""

DATADIR = op.join(op.dirname(__file__), "..", "data", "4a-english", "4A-English")
DATAFILE = op.join(DATADIR, "SemEval2017-task4-dev.subtask-A.english.INPUT.txt")
OUTFILE = op.join(DATADIR, "SemEval2017-task4-dev.subtask-A.english.INPUT_PREPROS.txt")

C = 0

def rept(matchobj):
    print("MATHCOBJ: " + matchobj.groups(0)[0])
    return str(matchobj.groups(0)[0] + matchobj.groups(0)[0])

with open(DATAFILE, "r") as reader:
    line = reader.readline()
    while line != None and C <= 100:
        line = re.sub(r'http:\/\/t\.co\/([A-Z0-9+&@#\/%=~_|$]*)',
                  "<URL>", line, flags=re.IGNORECASE)
        line = re.sub(r'([a-zA-Z])\1{2,}', rept, line)
        print(line)
        line = reader.readline()
        C += 1