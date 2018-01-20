from os import path as op

# http://help.sentiment140.com/for-students/
# https://www.crowdflower.com/data-for-everyone/
DOWNLOAD_SOURCES = [("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4a-english.zip", "4a-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4c-english.zip", "4c-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4e-english.zip", "4e-english-2017"),
            ("http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip", "sentiment140")]
TRAINING_TSV = "dataset_training.tsv"
VALIDATION_TSV = "dataset_validation.tsv"

def download_and_extract(url, fname):
    import urllib.request
    import zipfile

    tmp, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(tmp,"r") as zip_ref:
        zip_ref.extractall(fname)


def create_validation_tsv():
    import pandas

    taskA = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[0][1], "4A-English",
                            "SemEval2017-task4-dev.subtask-A.english.INPUT.txt"),
                            sep="\t", encoding="utf-8", names=["id", "semantic", "text", "empty"])
    taskA = taskA.drop(["empty"], axis=1)
    taskA["semantic"] = taskA["semantic"].map({"negative": -1,
                                               "neutral": 0,
                                               "positive": 1})

    taskC = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[1][1],
                                    "4C-English/SemEval2017-task4-dev.subtask-CE.english.INPUT.txt"),
                                    sep="\t", encoding="utf-8", names=["id", "name", "semantic", "text", "empty"])
    taskC = taskC.drop(["empty", "name"], axis=1)
    taskC["semantic"] = taskC["semantic"].replace({-2: -1,
                                                    2: 1})

    taskE = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[2][1],
                            "4E-English/SemEval2017-task4-dev.subtask-CE.english.INPUT.txt"),
                            sep="\t", encoding="utf-8", names=["id", "name", "semantic", "text", "empty"])
    taskE = taskE.drop(["empty", "name"], axis=1)
    taskE["semantic"] = taskE["semantic"].replace({-2: -1,
                                                    2: 1})

    total = pandas.concat([taskA, taskC, taskE])
    total = total.drop_duplicates(subset=["text"], keep='first')
    total.to_csv(op.join(op.dirname(__file__), VALIDATION_TSV), sep="\t", encoding="utf-8")


def create_training_tsv():
    import pandas

    stanf = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[3][1],
                            "testdata.manual.2009.06.14.csv"),
                            sep=",", encoding="latin-1", names=["semantic", "id", "date", "query", "user", "text"])
    print(stanf.columns)
    stanf = stanf.drop(["date", "query", "user"], axis=1)
    stanf["semantic"] = stanf["semantic"].replace({0: -1,
                                                   2: 0,
                                                   4: 1})
    stanf[["id", "semantic", "text"]].to_csv(op.join(op.dirname(__file__), TRAINING_TSV), sep="\t", encoding="utf-8")

if __name__ == "__main__":
    
    for t in DOWNLOAD_SOURCES:
        if not op.exists(op.join(op.dirname(__file__), t[1])):
            download_and_extract(t[0], op.join(op.dirname(__file__), t[1]))

    if not op.exists(op.join(op.dirname(__file__), VALIDATION_TSV)):
        create_validation_tsv()

    if not op.exists(op.join(op.dirname(__file__), TRAINING_TSV)):
        create_training_tsv()