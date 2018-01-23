from os import path as op

# http://help.sentiment140.com/for-students/
# https://www.crowdflower.com/data-for-everyone/
DOWNLOAD_SOURCES = [("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4a-english.zip", "4a-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4c-english.zip", "4c-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4e-english.zip", "4e-english-2017"),
            ("http://www.saifmohammad.com/WebDocs/stance-data-all-annotations.zip", "semeval-2016")]
TRAINING_CSV = "dataset_training.csv"
VALIDATION_CSV = "dataset_validation.csv"

def download_and_extract(url, fname):
    import urllib.request
    import zipfile

    tmp, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(tmp,"r") as zip_ref:
        zip_ref.extractall(fname)

def download_csv(url, fname):
    import urllib.request

    tmp, _ = urllib.request.urlretrieve(url, fname)

def create_validation_csv():
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

    combined = pandas.concat([taskA, taskC, taskE])
    combined = combined.drop_duplicates(subset=["text"], keep='first')
    combined.to_csv(op.join(op.dirname(__file__), VALIDATION_CSV), sep=",", encoding="utf-8")


def create_training_csv():
    import pandas

    taskA = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[3][1],
                            "data-all-annotations", "testdata-taskA-all-annotations.txt"),
                            sep="\t", encoding="utf-8", names=["id", "target", "text", "stance", "opinion", "semantic"],
                            skiprows=1)
    taskA = taskA[["id", "semantic", "text"]]
    taskA["semantic"] = taskA["semantic"].replace({"POSITIVE": 1,
                                                   "NEITHER": 0,
                                                   "NEGATIVE": -1})

    taskB = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[3][1],
                            "data-all-annotations", "testdata-taskB-all-annotations.txt"),
                            sep="\t", encoding="utf-8", names=["id", "target", "text", "stance", "opinion", "semantic"],
                            skiprows=1)
    taskB = taskB[["id", "semantic", "text"]]
    taskB["semantic"] = taskB["semantic"].replace({"POSITIVE": 1,
                                                   "NEITHER": 0,
                                                   "NEGATIVE": -1})

    # REMARK latin-1 encoding might cause trouble when saving to csv
    addTrain = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[3][1],
                            "data-all-annotations", "trainingdata-all-annotations.txt"),
                            sep="\t", encoding="latin-1", names=["id", "target", "text", "stance", "opinion", "semantic"],
                            skiprows=1)
    addTrain = addTrain[["id", "semantic", "text"]]
    addTrain["semantic"] = addTrain["semantic"].replace({"POSITIVE": 1,
                                                         "NEITHER": 0,
                                                         "NEGATIVE": -1})

    addTrial = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[3][1],
                            "data-all-annotations", "trialdata-all-annotations.txt"),
                            sep="\t", encoding="latin-1", names=["id", "target", "text", "stance", "opinion", "semantic"],
                            skiprows=1)
    addTrial = addTrial[["id", "semantic", "text"]]
    addTrial["semantic"] = addTrial["semantic"].replace({"POSITIVE": 1,
                                                         "NEITHER": 0,
                                                         "NEGATIVE": -1})

    combined = pandas.concat([taskA, taskB, addTrain, addTrial])
    combined = combined.drop_duplicates(subset=["text"], keep='first')
    combined.to_csv(op.join(op.dirname(__file__), TRAINING_CSV), sep=",", encoding="utf-8")


if __name__ == "__main__":
    
    # Download data contained in zip files
    for t in DOWNLOAD_SOURCES:
        folder_path = op.join(op.dirname(__file__), t[1])
        if not op.exists(folder_path):
            download_and_extract(t[0], folder_path)

    # Create validation data
    if not op.exists(op.join(op.dirname(__file__), VALIDATION_CSV)):
        create_validation_csv()

    # Create training data
    if not op.exists(op.join(op.dirname(__file__), TRAINING_CSV)):
        create_training_csv()