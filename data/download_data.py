from os import path as op

# http://help.sentiment140.com/for-students/
# https://www.crowdflower.com/data-for-everyone/
DOWNLOAD_SOURCES = [("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4a-english.zip", "4a-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4c-english.zip", "4c-english-2017"),
            ("http://alt.qcri.org/semeval2017/task4/data/uploads/codalab/4e-english.zip", "4e-english-2017"),
            ("http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip", "sentiment140"),
            ("https://github.com/zfz/twitter_corpus/archive/master.zip", "sanders-companies")]
ADDITIONAL_SOURCES = [("https://www.crowdflower.com/wp-content/uploads/2016/03/Apple-Twitter-Sentiment-DFE.csv", "apple_stock.csv")]
INCLUDED_DATA = [("us_airlines.zip", "us-airlines")]
TRAINING_TSV = "dataset_training.tsv"
VALIDATION_TSV = "dataset_validation.tsv"

def download_and_extract(url, fname):
    import urllib.request
    import zipfile

    tmp, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(tmp,"r") as zip_ref:
        zip_ref.extractall(fname)

def download_csv(url, fname):
    import urllib.request

    tmp, _ = urllib.request.urlretrieve(url, fname)

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

    combined = pandas.concat([taskA, taskC, taskE])
    combined = combined.drop_duplicates(subset=["text"], keep='first')
    combined.to_csv(op.join(op.dirname(__file__), VALIDATION_TSV), sep="\t", encoding="utf-8")


def create_training_tsv():
    import pandas

    stanf = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[3][1],
                            "testdata.manual.2009.06.14.csv"),
                            sep=",", encoding="latin-1", names=["semantic", "id", "date", "query", "user", "text"])
    
    stanf = stanf.drop(["date", "query", "user"], axis=1)
    stanf["semantic"] = stanf["semantic"].replace({0: -1,
                                                   2: 0,
                                                   4: 1})
    
    sanders = pandas.read_csv(op.join(op.dirname(__file__), DOWNLOAD_SOURCES[4][1],
                            "twitter_corpus-master", "full-corpus.csv"),
                            sep=",", encoding="utf-8", names=["topic", "semantic", "id", "date", "text"], skiprows=1)
    sanders = sanders.drop(["date", "topic"], axis=1)
    sanders = sanders[sanders["semantic"] != "irrelevant"]
    sanders["semantic"] = sanders["semantic"].replace({"positive": 1,
                                                       "neutral": 0,
                                                       "negative": -1})

    apple = pandas.read_csv(op.join(op.dirname(__file__), "additional_sources", ADDITIONAL_SOURCES[0][1]),
                            sep=",", encoding="latin-1", names=["id", "golden", "state", "trusted", "last_judge", "semantic",
                                                              "confidence", "date", "drop_id", "query", "semantic_gold", "text"], skiprows=1)
    apple = apple[["id", "semantic", "text"]]
    apple = apple[apple["semantic"] != "not_relevant"]
    apple = apple["semantic"].replace({1: -1,
                                       3: 0,
                                       5: 1})

    airline = pandas.read_csv(op.join(op.dirname(__file__), INCLUDED_DATA[0][1], "Tweets.csv"),
                            sep=",", encoding="utf-8", names=["id", "semantic", "semantic_confidence", "neg_reason", "reason_conf",
                                                              "airline", "airline_sentiment_gold", "user", "reason_gold", "retweet_count",
                                                              "text", "tweet_coord", "date", "tweet_location", "user_timezone"], skiprows=1)
    airline = airline[["id", "semantic", "text"]]
    airline["semantic"] = airline["semantic"].replace({"negative": -1,
                                                       "neutral": 0,
                                                       "positive": 1})

    combined = pandas.concat([stanf[["id", "semantic", "text"]],
                              sanders[["id", "semantic", "text"]],
                              apple,
                              airline])
    combined = combined.drop_duplicates(subset=["text"], keep='first')
    combined.to_csv(op.join(op.dirname(__file__), TRAINING_TSV), sep="\t", encoding="utf-8")

if __name__ == "__main__":
    
    # Download data contained in zip files
    for t in DOWNLOAD_SOURCES:
        folder_path = op.join(op.dirname(__file__), t[1])
        if not op.exists(folder_path):
            download_and_extract(t[0], folder_path)

    # Create folder for additional sources ( not in zip )
    if not op.exists(op.join(op.dirname(__file__), "additional_sources")):
        from os import mkdir
        mkdir(op.join(op.dirname(__file__), "additional_sources"))

    # Download and save csv data sources
    for t in ADDITIONAL_SOURCES:
        csv_path = op.join(op.dirname(__file__), "additional_sources", t[1])
        if not op.exists(csv_path):
            download_csv(t[0], csv_path)

    # Extract predistributed zips
    for t in INCLUDED_DATA:
        zip_path = op.join(op.dirname(__file__), "included_data", t[0])
        folder_path = op.join(op.dirname(__file__), t[1])
        if not op.exists(folder_path):
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(folder_path)

    # Create validation data
    if not op.exists(op.join(op.dirname(__file__), VALIDATION_TSV)):
        create_validation_tsv()

    # Create training data
    if not op.exists(op.join(op.dirname(__file__), TRAINING_TSV)):
        create_training_tsv()