from os import path as op
from matplotlib import pyplot

def parseFromSemEval(file):
    import pandas
    # This should be wrapped in check
    f = pandas.read_csv(file, sep=",", encoding="utf-8", names=["id", "semantic", "text"], skiprows=1, index_col=1)
    return f[["text", "semantic"]].as_matrix()

if __name__ == "__main__":
    import numpy as np
    DATA = parseFromSemEval(op.join(op.dirname(__file__), "dataset_validation.csv"))[:,1]

    neg = np.sum(DATA == -1)
    neu = np.sum(DATA == 0)
    pos = np.sum(DATA == 1)

    print("===== VALIDATION =====")
    print("NEG percentage {}".format(float(neg)/len(DATA)))
    print("NEU percentage {}".format(float(neu)/len(DATA)))
    print("POS percentage {}".format(float(pos)/len(DATA)))

    DATA = parseFromSemEval(op.join(op.dirname(__file__), "dataset_training.csv"))[:,1]

    neg = np.sum(DATA == -1)
    neu = np.sum(DATA == 0)
    pos = np.sum(DATA == 1)

    print("====== TRAINING ======")
    print("NEG percentage {}".format(float(neg)/len(DATA)))
    print("NEU percentage {}".format(float(neu)/len(DATA)))
    print("POS percentage {}".format(float(pos)/len(DATA)))