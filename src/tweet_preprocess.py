from os import path as op
import re
import pandas


DATADIR = op.join(op.dirname(__file__), "..", "data")
DATAFILE = op.join(DATADIR, "4a-english", "4A-English", "SemEval2017-task4-dev.subtask-A.english.INPUT.txt")
OUTFILE = op.join(DATADIR, "4a-english", "4A-English", "SemEval2017-task4-dev.subtask-A.english.INPUT_PREPROS.txt")

EMOJIS = op.join(DATADIR, "emojis.csv")
EMOJIS = pandas.read_csv(EMOJIS, sep=",", index_col=0, encoding="utf-8")

C = 0

def rept(matchobj):
    return str(matchobj.groups(0)[0] + matchobj.groups(0)[0])

def clicheLine(line):
    # Notes from Mathieu Cliche https://arxiv.org/pdf/1704.06125.pdf,
    # Winner of the SemEval 2017:
    """
        - URLS replaced by <url>
        - Emojis replaced by e.g <smile>, <sadface>, <lolface>, <neutralface>
        - Any letter repeated more than 2 times in a row, replaced by 2 repetitions. E.g "cooool" -> "cool"
        - All tweets lowercased
        
        returns: String
    """
    line = line.lower()
    line = re.sub(r'http:\/\/t\.co\/([A-Z0-9+&@#\/%=~_|$]*)',
              "<URL>", line, flags=re.IGNORECASE)
    line = re.sub(r'([a-zA-Z])\1{2,}', rept, line)
    return line


with open(DATAFILE, "r") as reader:
    line = reader.readline()
    while line != None and C <= 100:
        line = clicheLine(line)
        print(line)
        line = reader.readline()
        C += 1

print(EMOJIS)