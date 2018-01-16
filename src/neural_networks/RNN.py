import torch
import torch.nn as nn
import numpy as np
import json
import shutil
import torch.nn.functional as F
from torch.autograd import Variable
from os import path as op
from sklearn.model_selection import train_test_split

MAX_LEN = 140       # Lenth of a tweet
BATCH_SIZE = 256
EPOCH = 25         # With epoch 0, we will run until interrupted
LR = 1e-5
CONTINUE = True     # Attempts to continue from previous checkpoint
DEBUG = False
CUDA = True
DATA_SLICE = 40000

CHECKPOINT_PATH = op.join(op.dirname(__file__), "..", "..", "checkpoint.tar")
MODEL_PATH = op.join(op.dirname(__file__), "..", "..", "model.tar")


def parseFromSemEval(file):
    import pandas
    
    f = pandas.read_csv(file, sep="\t", encoding="utf-8", index_col=0)
    return f[["text", "semantic"]].as_matrix()


def batch(tensor, batch_size):
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
        self.validation_acc = []

        for t in range(1, nb_epoch + 1):
            loss, acc = self._fit(X_list, y_list)
            self.training_cost.append(loss)
            self.training_acc.append(acc)
            val_log = ''
            if validation_data:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1], batch_size)
                val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
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
        # Original y_pred = self.model(X, self.model.initHidden(X.size()[1]))
        # Init hidden 100, as we perform embedding in the GRU
        y_pred = self.model(X, init_hidden)
        return y_pred		

    def predict_classes(self, X):
        return torch.topk(self.predict(X), 1)[1].cpu().data.numpy().flatten()


#############


class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, weights_path=None, dict_path=None):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(401,embed_size)

        self.rnn = nn.GRU(embed_size, hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, output_size)

        if weights_path is not None:
            self._load_weights(weights_path)

        self.softmax = nn.LogSoftmax(dim=1)

    def _load_weights(self, weights_path):
        """ Only works with original weights"""
        import h5py
        H5 = h5py.File(weights_path, "r")
        self.embed.weight = nn.Parameter(torch.from_numpy(H5['embedding/embedding/embeddings'].value), requires_grad=False)
        # Saved files have axes swapped
        self.rnn.weight_ih = nn.Parameter(torch.from_numpy(np.transpose(H5['rnn/rnn/kernel'].value)), requires_grad=False)
        self.rnn.weight_hh = nn.Parameter(torch.from_numpy(np.transpose(H5['rnn/rnn/recurrent_kernel'].value)), requires_grad=False)
        
        self.rnn.bias_ih = nn.Parameter(torch.from_numpy(H5['rnn/rnn/bias'].value), requires_grad=False)
        self.rnn.bias_hh = nn.Parameter(torch.from_numpy(H5['rnn/rnn/bias'].value), requires_grad=False)
        
        # Only train output layer
        self.output.weight = nn.init.xavier_normal(nn.Parameter(torch.zeros(3,256), requires_grad=True))
        self.output.bias_ih = nn.init.xavier_normal(nn.Parameter(torch.zeros(3,256), requires_grad=True))


    def forward(self, input, hidden):
        embedded = self.embed(input)
        embedded.transpose_(0,1)

        out, hidden = self.rnn(embedded, hidden)
        lin = F.relu(self.output(out[MAX_LEN-1,:,:]))


        return lin, hidden

    def initHidden(self, batch_size, input_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_size))



def main():

    DATADIR = op.join(op.dirname(__file__), "..", "..", "data")
    DATAFILE = op.join(DATADIR, "semeval-2017.tsv")
    if not op.exists(DATAFILE):
        print("Could not find semeval-2017.tsv file. Please run prep_semeval.py from data directory")
        return -1
    VOCAB = op.join(op.dirname(__file__), "..", "..", "reactionrnn_pretrained", "reactionrnn_vocab.json")
    CONVERT_TABLE = json.load(open(VOCAB))
    DATA = parseFromSemEval(DATAFILE)
    
    # Convert according to VOCAB
    CONVERTED = np.zeros((DATA.shape[0], 140))
    for i in range(DATA.shape[0]):
        txt = DATA[i,0]
        for j in range(min(len(txt), 140)):
            try:
                CONVERTED[i,j] = CONVERT_TABLE[txt[j]]
            except KeyError:
                # Keep as 0
                pass
            

    X = CONVERTED
    y = DATA[:,1].astype(int)
    indices = np.random.permutation(X.shape[0])
    X = X[indices,:]
    y = y[indices]

    if DEBUG:
        X_train, X_test, y_train, y_test = train_test_split(X[:5000,:], y[:5000], test_size=.2)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X[:DATA_SLICE,:], y[:DATA_SLICE], test_size=.2)

    epoch = 0
    best_prec = 0.0
    training_cost = []
    training_acc = []
    validation_acc = []

    model = RNN(140, 100, 256, 3, weights_path=op.join(op.dirname(__file__), "..", "..", "reactionrnn_pretrained", "reactionrnn_weights.hdf5"))
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
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        training_cost = checkpoint["train_cost"]
        training_acc = checkpoint["train_hist"]
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

    clf = Estimator(model)
    clf.compile(optimizer,
                loss=nn.CrossEntropyLoss())
    
    try:
        if EPOCH == 0:
            c = 0
            while True:
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
            'valid_hist':   validation_acc
        }, is_best)

if __name__ == "__main__":
    main()
