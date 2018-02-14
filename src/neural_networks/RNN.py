import torch
import torch.nn as nn
import numpy as np
import json
import shutil
import torch.nn.functional as F
from torch.autograd import Variable
from os import path as op
from sklearn.model_selection import train_test_split
from matplotlib import pyplot, patches

MAX_LEN = 140       # Lenth of a tweet
BATCH_SIZE = 512
EPOCH = 250         # With epoch 0, we will run until interrupted
LR = 1e-4           # LR 1e-4 seems to give stable learning without big oscillation
CONTINUE = True     # Attempts to continue from previous checkpoint
DEBUG = False
CUDA = True
TEST_WITH_VALIDATION = False    # Only test with validation data
DATA_SLICE = 40000

CHECKPOINT_PATH = op.join(op.dirname(__file__), "..", "..", "checkpoint.pt")
MODEL_PATH = op.join(op.dirname(__file__), "..", "..", "model.pt")


def parseFromSemEval(file):
    # TODO Move to utils
    # TODO Remove dependency on Pandas
    import pandas
    
    f = pandas.read_csv(file, sep=",", encoding="utf-8", index_col=0)
    return f[["text", "semantic"]].as_matrix()

def _convert_with_vocab(data, vocab_table):
    # Convert according to VOCAB
    # TODO Might not work if shape is only 1-d.
    CONVERTED = np.zeros((data.shape[0], 140))
    for i in range(data.shape[0]):
        txt = data[i,0]
        for j in range(min(len(txt), 140)):
            try:
                CONVERTED[i,j] = vocab_table[txt[j]]
            except KeyError:
                # Keep as 0
                pass
    return CONVERTED

def _loadSemEvalData(fname):
    """
    Load data from predefined SemEval sources.
    
        Returns: (Training-data, Training-labels, Validation-data, Validation-labels)
    """
    
    DATADIR = op.join(op.dirname(__file__), "..", "..", "data")

    # Test if files exist
    if not op.exists(fname):
        # Check alternative path
        if not op.exists(op.join(DATADIR, fname)):
            print("Could not find {} file. Please run download_data.py from data directory".format(op.join(DATADIR, fname)))
            return 0
        else:
            fname = op.join(DATADIR, fname)
        
    data = parseFromSemEval(fname)
    return data

def _loadCharacterEmbedding():
    """
    Load character-embedding indexes.

        Returns: dict(character, index)
    """
    # Path to unpacked file
    # TODO For packaging use path to site
    VOCAB = op.join(op.dirname(__file__), "..", "..", "assets", "embeddings", "reactionrnn_vocab.json")

    if not op.exists(VOCAB):
        print("Fatal error")
        print("Could not find {} file. Has it been deleted? Can be downloaded from https://github.com/Manezki/TwitMine/blob/master/assets/embeddings/reactionrnn_vocab.json".format(VOCAB))
        sys.exit(-1)

    CONVERT_TABLE = json.load(open(VOCAB))
    return CONVERT_TABLE

def batch(tensor, batch_size):
    # TODO Move to utils
    # TODO Change to be more concervative with memory
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

def save_checkpoint(state, is_best, filename=CHECKPOINT_PATH):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, MODEL_PATH)


def plot_progress(training, validation, loss=False):
    # TODO move to utils
    xaxis = np.linspace(1, 1+len(training), num=len(training))
    pl1 = pyplot.plot(xaxis, training, color='orange')
    pl2 = pyplot.plot(xaxis, validation, color='blue')
    if not loss:
        pyplot.title("Training vs Validation accuracy")
        pyplot.xlabel("Epoch")
        pyplot.ylabel("Accuracy (%)")
        orange = patches.Patch(color='orange', label="Training accuracy")
        blue = patches.Patch(color='blue', label="Validation accuracy")
    else:
        minIdx = np.argmin(validation)
        miny = np.min(training)
        pyplot.plot([minIdx, minIdx+1], [miny, validation[minIdx]], color="red")
        pyplot.title("Training vs Validation loss")
        pyplot.xlabel("Epoch")
        pyplot.ylabel("Loss")
        orange = patches.Patch(color='orange', label="Training loss")
        blue = patches.Patch(color='blue', label="Validation loss")
    pyplot.legend(handles=[orange, blue])
    pyplot.show()


class Estimator(object):
    ## Based on woderfull Gist https://gist.github.com/kenzotakahashi/ed9631f151710c6bd898499fcf938425

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, X_list, y_list):
        """
        train one epoch
        """
        loss_list = []
        acc_list = []

        for X, y in zip(X_list, y_list):
            if CUDA:
                X_v = Variable(torch.from_numpy(X).long(), requires_grad=False).cuda()
                y_v = Variable(torch.from_numpy(y + 1).long(), requires_grad=False).cuda()
                init_hidden = self.model.initHidden(X.shape[0], 100).cuda()
            else:
                X_v = Variable(torch.from_numpy(X).long(), requires_grad=False)
                y_v = Variable(torch.from_numpy(y + 1).long(), requires_grad=False)
                init_hidden = self.model.initHidden(X.shape[0], 100)
            

            self.optimizer.zero_grad()
            # Original y_pred = self.model(X, self.model.initHidden(X.size()[1]))
            # Init hidden 100, as we perform embedding in the GRU
            y_pred, hidden = self.model(X_v, init_hidden)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.data[0])
            classes = torch.topk(y_pred, 1)[1].cpu().data.numpy().flatten()
            #comp = np.hstack((classes.reshape(-1,1), (y+1).reshape(-1,1)))
            #print(comp)
            
            acc = self._accuracy(classes, y+1)
            acc_list.append(acc)

        return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        # TODO keep track of the best model state and return it when finished
        X_list = batch(X, batch_size)
        y_list = batch(y, batch_size)

        self.training_cost = []
        self.training_acc = []
        self.validation_cost = []
        self.validation_acc = []

        for t in range(1, nb_epoch + 1):
            loss, acc = self._fit(X_list, y_list)
            self.training_cost.append(loss)
            self.training_acc.append(acc)
            val_log = ''
            if validation_data:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1], batch_size)
                val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
                self.validation_cost.append(val_loss)
                self.validation_acc.append(val_acc)
            print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))
            
    def evaluate(self, X, y, batch_size=32):
        y_pred, hidden = self.predict(X)

        if CUDA:
            y_v = Variable(torch.from_numpy(y + 1).long(), requires_grad=False).cuda()
        else:
            y_v = Variable(torch.from_numpy(y + 1).long(), requires_grad=False)
        loss = self.loss_f(y_pred, y_v)

        classes = torch.topk(y_pred, 1)[1].cpu().data.numpy().flatten()
        acc = self._accuracy(classes, y+1)
        
        _, gt = np.unique(y + 1, return_counts=True)
        gt = gt.astype(float) / len(y)
        _, pr = np.unique(classes, return_counts=True)
        pr = pr.astype(float) / len(y)
        if len(gt) == 3 and len(pr) == 3:
            print("Distribution Grund truth: NEG {}, NEU {}, POS {}".format(gt[0], gt[1], gt[2]))
            print("Distribution predictions: NEG {}, NEU {}, POS {}".format(pr[0], pr[1], pr[2]))

        return loss.data[0], acc

    def _accuracy(self, y_pred, y):
        return sum(y_pred == y) / y.shape[0]

    def predict(self, X):
        if CUDA:
            X = Variable(torch.from_numpy(X).long()).cuda()
            init_hidden = self.model.initHidden(X.shape[0], 100).cuda()
        else:
            X = Variable(torch.from_numpy(X).long())
            init_hidden = self.model.initHidden(X.shape[0], 100)
        y_pred = self.model(X, init_hidden)
        return y_pred		

    def predict_classes(self, X):
        return torch.topk(self.predict(X), 1)[1].cpu().data.numpy().flatten()


#############

class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, state_dict=None, dict_path=None):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(401,embed_size, padding_idx=0)

        self.rnn = nn.GRU(embed_size, hidden_size, bias=True, dropout=0.5)
        self.output = nn.Linear(hidden_size, output_size)

        self._create_weight_tensors(input_size, hidden_size, output_size)

        if state_dict is not None:
            self._load_weights(state_dict)
        else:
            self._init_weights(nn.init.kaiming_normal)

        self.softmax = nn.LogSoftmax(dim=1)

    def _load_weights(self, state_dict):
        pretrained = torch.load(state_dict)
        self.load_state_dict(pretrained['state_dict'])

    def _create_weight_tensors(self, input_size, hidden_size, output_size):
        self.embed.weight = nn.Parameter(torch.zeros(401, 100))
        self.rnn.weight_ih = nn.Parameter(torch.zeros(3*hidden_size, 100))
        self.rnn.weight_hh = nn.Parameter(torch.zeros(3*hidden_size, hidden_size))
        self.rnn.bias_ih = nn.Parameter(torch.zeros(3*hidden_size))
        self.rnn.bias_hh = nn.Parameter(torch.zeros(3*hidden_size))
        self.output.weight = nn.Parameter(torch.zeros(3, 256))
        self.output.bias_ih = nn.Parameter(torch.zeros(3, 256))

    def _init_weights(self, method):
        method(self.embed.weight)
        method(self.rnn.weight_ih)
        method(self.rnn.weight_hh)
        method(self.output.weight)
        # Bias already 0s

    def forward(self, input, hidden):
        embedded = self.embed(input)
        embedded.transpose_(0,1)

        out, hidden = self.rnn(embedded, hidden)
        lin = F.relu(self.output(out[MAX_LEN-1,:,:]))

        return lin, hidden

    def initHidden(self, batch_size, input_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size))



def main():

    training = _loadSemEvalData("dataset_training.csv")
    validation = _loadSemEvalData("dataset_validation.csv")

    # This line prevents running if the data was not loaded, refrase the check for more specific use.
    # Training and Validation should be int only when bad loading
    if isinstance(training, int) and isinstance(validation, int):
        sys.exit(-1)

    # If DATASLICE is smaller than data amount, take a subset.
    training   = training[:DATA_SLICE, :]
    validation = validation[:DATA_SLICE, :]

    # Convert text column to embedding indexes
    CONVERT_TABLE = _loadCharacterEmbedding()
    training_data   = _convert_with_vocab(training, CONVERT_TABLE)
    validation_data = _convert_with_vocab(validation, CONVERT_TABLE)

    training_labels   = training[:, 1].astype(int)
    validation_labels = validation[:, 1].astype(int)

    # Split the training data to test and training set.
    # Holdout-method is used, and no further cross validation is performed.
    # TODO Change naming convention from Training, test, validation(unseen data) to Training, validation, test
    X_train = training_data[:int(training_data.shape[0]*0.8), :]
    X_test  = training_data[int(training_data.shape[0]*0.8):, :]
    y_train = training_labels[:int(training_labels.shape[0]*0.8)]
    y_test  = training_labels[int(training_labels.shape[0]*0.8):]

    epoch = 0
    best_prec = 0.0
    training_cost = []
    training_acc = []
    validation_cost = []
    validation_acc = []

    model = RNN(140, 100, 256, 3, state_dict=op.join(op.dirname(__file__), "..", "..", "assets", "weights", "RNN.pt"))
    if torch.cuda.is_available() and CUDA:
        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        
    if op.exists(MODEL_PATH) and CONTINUE:
        # TODO, cannot continue if was trained on CPU and continues on GPU and vice versa
        print("Continuying with the previous model")
        checkpoint = torch.load(MODEL_PATH)
        epoch = checkpoint["epoch"]
        best_prec = checkpoint["best_prec"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        for paramGroup in optimizer.param_groups:
            paramGroup['lr'] = LR
        training_cost = checkpoint["train_cost"]
        training_acc = checkpoint["train_hist"]
        validation_cost = checkpoint['valid_cost']
        validation_acc = checkpoint["valid_hist"]
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))


    print(model)


    def fit_and_log(epoch):
        clf.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=epoch,
            validation_data=(X_test, y_test))
        [training_cost.append(i) for i in clf.training_cost]
        [training_acc.append(i) for i in clf.training_acc]
        [validation_acc.append(i) for i in clf.validation_acc]
        [validation_cost.append(i) for i in clf.validation_cost]

    clf = Estimator(model)
    clf.compile(optimizer,
                loss=nn.CrossEntropyLoss())
                #loss=nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([2,1,1.5])))
    
    if TEST_WITH_VALIDATION:
        _, VAL_ACC = clf.evaluate(VALIDATION_DATA, VALIDATION_Y, BATCH_SIZE)
        print("Validation accuracy on the unseen validation data {}".format(VAL_ACC))
        plot_progress(training_acc, validation_acc)
        plot_progress(training_cost, validation_cost, loss=True)
        return -1
    try:
        if EPOCH == 0:
            c = 0
            while True:
                # TODO only saves after finished, should keep tract of the best weights.
                print("Training epoch: {} from current run".format(c))
                fit_and_log(1)
                c+=1
                epoch += 1
        else:
            fit_and_log(EPOCH)
            epoch += EPOCH
    except (KeyboardInterrupt, SystemExit):
        # Save the model
        if len(validation_acc) != 0:
            is_best = validation_acc[-1] > best_prec
            best_prec = max(validation_acc[-1], best_prec) 
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict':   model.state_dict(),
                    'best_prec':    best_prec,
                    'optimizer':    optimizer.state_dict(),
                    'train_cost':   training_cost,
                    'train_hist':   training_acc,
                    'valid_cost':   valid_cost,
                    'valid_hist':   validation_acc
                }, is_best)
        print("Saved model after interrupt")
        raise    

    score, acc = clf.evaluate(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    # Save the model
    is_best = acc > best_prec
    best_prec = max(acc, best_prec) 
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict':   model.state_dict(),
            'best_prec':    best_prec,
            'optimizer':    optimizer.state_dict(),
            'train_cost':   training_cost,
            'train_hist':   training_acc,
            'valid_cost':   validation_cost,
            'valid_hist':   validation_acc
        }, is_best)

    _, VAL_ACC = clf.evaluate(VALIDATION_DATA, VALIDATION_Y, BATCH_SIZE)
    print("Validation accuracy on the unseen validation data {}".format(VAL_ACC))

    plot_progress(training_acc, validation_acc)
    plot_progress(training_cost, validation_cost, loss=True)

if __name__ == "__main__":
    main()