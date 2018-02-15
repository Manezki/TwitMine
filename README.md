# TwitMine
As more and more people are creating content on Internet, the amount of available data is growing fast.<br>
Solely Twitter is producing [500 million tweets per day](http://www.internetlivestats.com/twitter-statistics/) which can provide valuable insights to the overall opinions.<br>
<br>
For this purpose, TwitMine for generated. Currently package contains RNN that classifies tweets
into "positive", "neutral" and "negative" labels.
What sets this apart from the others projects, we are processing tweets charcter by character - because "GREAT" is different than "great". <br>
<br>
Our training data has good quality (from [SemEval competition](http://alt.qcri.org/semeval2018/index.php?id=tasks)), but the dataset is small (check "additional-data" branch for more sources) - but with the dataset that we currently have in use, our average accuracy for correct label is ~60%.

# Requirements

```
Python3.*
PyTorch
```

Pytorch is available for download from their [homepage](http://pytorch.org/).

```
Additional requirements listed in 'requirements.txt'
```

# Contribution

Pull Requests are always welcome, and greatly appreciated.<br>
Even some basics properties could use help, like:
 * Including the data from SemEval competitions (2013-2015).
 * Reformatting the code to importable form.
 * Better handling of memory during the training.

If you happen to have access to labeled tweets, that could be added to the repertoire, please drop a issue :)


# Additional Credits
https://github.com/minimaxir/reactionrnn<br>
Model is retrained from his package.

http://alt.qcri.org/semeval2018/index.php?id=tasks<br>
For pushing the task forward and releasing datasets for open use.
