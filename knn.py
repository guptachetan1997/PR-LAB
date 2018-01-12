import numpy as np
import pandas as pd
from collections import Counter

class KNN():
	def __init__(self, X, y, k, split=0.6):
		print("KNN classifier initialsed.")
		self.X = X
		self.y = y
		self.k = k
		self.split = split
		self.train_test_split()
	
	def train_test_split(self):
		limit = int(self.split*len(self.X))
		self.trainX, self.trainY = self.X[:limit, :], self.y[:limit]
		self.testX, self.testY = self.X[limit:, :], self.y[limit:]
	
	def ed(self, v1, v2):
		return np.sqrt(np.dot(v1-v2, v1-v2))
	
	def getNeighboursVote(self):
		self.distances = sorted(self.distances, key = lambda k : k[0])[:self.k]
		c = Counter([_class[1] for _class in self.distances])
		return c.most_common()[0][0]

	def predict(self, inputVector):
		self.distances = [(self.ed(inputVector, x), y) for x,y in zip(self.trainX, self.trainY)]
		vote = self.getNeighboursVote()
		return vote
	
	def test(self):
		correct = 0
		for inputVector,label in zip(self.testX, self.testY):
			vote = self.predict(inputVector)
			if vote == label:
				correct += 1
		print("Accuracy is {}".format(100*correct/len(self.testX)))

def main():
	df = pd.read_csv('iris.data')
	df = df.sample(frac=1).values
	x = df[:, :4]
	y = df[:, -1]
	knn = KNN(x, y, 3)
	knn.test()

if __name__ == '__main__':
	main()

# OUTPUT:
# KNN classifier initialsed.
# Accuracy is 96.66666666666667