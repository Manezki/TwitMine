from os import path as op
from matplotlib import pyplot

def parseFromSemEval(file):
    import pandas
    # This should be wrapped in check
    f = pandas.read_csv(file, sep="\t", encoding="utf-8", names=["id", "semantic", "text", "empty"])
    f = f.drop(["empty"], axis=1)
    f["semantic"] = f["semantic"].map({"negative": -1,
                                       "neutral": 0,
                                       "positive": 1})
    return f[["text", "semantic"]].as_matrix()

if __name__ == "__main__":
    import numpy as np
    DATA = parseFromSemEval(op.join(op.dirname(__file__), "4a-english", "4A-English", "SemEval2017-task4-dev.subtask-A.english.INPUT.txt"))[:,1]

    neg = np.sum(DATA == -1)
    neu = np.sum(DATA == 0)
    pos = np.sum(DATA == 1)

    print("NEG percentage {}".format(float(neg)/len(DATA)))         #0.15660
    print("NEU percentage {}".format(float(neu)/len(DATA)))         #0.50126
    print("POS percentage {}".format(float(pos)/len(DATA)))         #0.34214