from os import path as op

SOURCES = [("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4a-english.zip", "4a-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4c-english.zip", "4c-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4e-english.zip", "4e-english-2017")]


def download_and_extract(url, fname):
    import urllib.request
    import zipfile

    tmp, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(tmp,"r") as zip_ref:
        zip_ref.extractall(fname)

def refine():
    import pandas

    taskA = pandas.read_csv(op.join(op.dirname(__file__), SOURCES[0][1], "4A-English",
                            "SemEval2017-task4-dev.subtask-A.english.INPUT.txt"),
                            sep="\t", encoding="utf-8", names=["id", "semantic", "text", "empty"])
    taskA = taskA.drop(["empty"], axis=1)
    taskA["semantic"] = taskA["semantic"].map({"negative": -1,
                                               "neutral": 0,
                                               "positive": 1})

    taskC = pandas.read_csv(op.join(op.dirname(__file__), SOURCES[1][1],
                                    "4C-English/SemEval2017-task4-dev.subtask-CE.english.INPUT.txt"),
                                    sep="\t", encoding="utf-8", names=["id", "name", "semantic", "text", "empty"])
    taskC = taskC.drop(["empty", "name"], axis=1)
    taskC["semantic"] = taskC["semantic"].replace({-2: -1,
                                                    2: 1})

    taskE = pandas.read_csv(op.join(op.dirname(__file__), SOURCES[2][1],
                            "4E-English/SemEval2017-task4-dev.subtask-CE.english.INPUT.txt"),
                            sep="\t", encoding="utf-8", names=["id", "name", "semantic", "text", "empty"])
    taskE = taskE.drop(["empty", "name"], axis=1)
    taskE["semantic"] = taskE["semantic"].replace({-2: -1,
                                                    2: 1})

    total = pandas.concat([taskA, taskC, taskE])
    total.to_csv(op.join(op.dirname(__file__), "semeval-2017.tsv"), sep="\t", encoding="utf-8")


if __name__ == "__main__":
    
    for t in SOURCES:
        if not op.exists(op.join(op.dirname(__file__), t[1])):
            download_and_extract(t[0], op.join(op.dirname(__file__), t[1]))

    if not op.exists(op.join(op.dirname(__file__), "semeval-2017.tsv")):
        refine()