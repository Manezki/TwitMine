import torch
import torch.nn as nn
import numpy as np
import json
import torch.nn.functional as F
from torch.autograd import Variable
from os import path as op



MAX_LEN = 30
EMBEDDING_SIZE = 64
BATCH_SIZE = 32
EPOCH = 40
DATA_SIZE = 1000
INPUT_SIZE = 300


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
			X_v = Variable(torch.from_numpy(np.swapaxes(X,0,1)).float())
			y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)

			self.optimizer.zero_grad()
			y_pred = self.model(X_v, self.model.initHidden(X_v.size()[1]))
			loss = self.loss_f(y_pred, y_v)
			loss.backward()
			self.optimizer.step()

			## for log
			loss_list.append(loss.data[0])
			classes = torch.topk(y_pred, 1)[1].data.numpy().flatten()
			acc = self._accuracy(classes, y)
			acc_list.append(acc)

		return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)

	def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
		X_list = batch(X, batch_size)
		y_list = batch(y, batch_size)

		for t in range(1, nb_epoch + 1):
			loss, acc = self._fit(X_list, y_list)
			val_log = ''
			if validation_data:
				val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1], batch_size)
				val_log = "- val_loss: %06.4f - val_acc: %06.4f" % (val_loss, val_acc)
			print("Epoch %s/%s loss: %06.4f - acc: %06.4f %s" % (t, nb_epoch, loss, acc, val_log))

	def evaluate(self, X, y, batch_size=32):
		y_pred = self.predict(X)

		y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
		loss = self.loss_f(y_pred, y_v)

		classes = torch.topk(y_pred, 1)[1].data.numpy().flatten()
		acc = self._accuracy(classes, y)
		return loss.data[0], acc

	def _accuracy(self, y_pred, y):
		return sum(y_pred == y) / y.shape[0]

	def predict(self, X):
		X = Variable(torch.from_numpy(np.swapaxes(X,0,1)).float())		
		y_pred = self.model(X, self.model.initHidden(X.size()[1]))
		return y_pred		

	def predict_classes(self, X):
		return torch.topk(self.predict(X), 1)[1].data.numpy().flatten()


#############


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, output_size, weights_path=None, dict_path=None):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(401,embed_size)

        self.rnn = nn.GRUCell(embed_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        if weights_path is not None:
            self._load_weights(weights_path)

        self.softmax = nn.LogSoftmax(dim=1)

    def _load_weights(self, weights_path):
        import h5py
        # op.join(op.dirname(__file__), "..", "..", "reactionrnn", "reactionrnn", "reactionrnn_weights.hdf5")
        H5 = h5py.File(weights_path, "r")
        self.embed.weight = nn.Parameter(torch.from_numpy(H5['embedding/embedding/embeddings'].value))
        self.rnn.weight_ih = nn.Parameter(torch.from_numpy(H5['rnn/rnn/kernel'].value))
        self.rnn.weight_hh = nn.Parameter(torch.from_numpy(H5['rnn/rnn/recurrent_kernel'].value))
        self.rnn.bias_ih = nn.Parameter(torch.from_numpy(H5['rnn/rnn/bias'].value))
        self.output.weight = nn.Parameter(torch.from_numpy(H5['output/output/kernel'].value))
        self.output.bias_ih = nn.Parameter(torch.from_numpy(H5['output/output/bias'].value))


    def forward(self, input, hidden):
        hidden = Variable(torch.zeros(1, self.hidden_size))

        for i in input:
            _, hidden = self.rnn(self.embed(input(i)), hidden)

        output = self.softmax(self.output(hidden))
        return output, hidden

RNN(140,256,100,3, weights_path=op.join(op.dirname(__file__), "..", "..", "reactionrnn", "reactionrnn", "reactionrnn_weights.hdf5"))

################################



def main():


	## Fake data
	X = np.random.randn(DATA_SIZE * 1, MAX_LEN, INPUT_SIZE)
	y = np.array([i for i in range(1) for _ in range(DATA_SIZE)])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

	model = GRU(INPUT_SIZE, EMBEDDING_SIZE, 1)
	clf = Estimator(model)
	clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
				loss=nn.CrossEntropyLoss())
	clf.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH,
			validation_data=(X_test, y_test))
	score, acc = clf.evaluate(X_test, y_test)
	print('Test score:', score)
	print('Test accuracy:', acc)

	torch.save(model, 'model.pt')

if __name__ == "__main__":
	main()