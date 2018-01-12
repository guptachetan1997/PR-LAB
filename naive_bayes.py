import numpy as np
import pandas as pd
from collections import Counter

class NaiveBayes():
	
	def __init__(self, x, y, class_count, split=0.6):
		self.data = x
		self.labelName = y
		self.class_count = class_count
		self.split = split
		self.split_data()
	
	def split_data(self):
		print("Train text split is {}".format(self.split))
		limit = int(self.split*self.data.shape[0])
		self.train_data, self.train_labels = self.data.iloc[:limit , :], self.data.iloc[:limit, -1]
		self.test_data, self.test_labels = self.data.iloc[limit: , :], self.data.iloc[limit:, -1]

	def calcProb(self, x, mean, std):
		return (1/(np.sqrt(2*np.pi)*std))*np.exp((-1/2)*(((x - mean)/std)**2))
	
	def separateClasses(self):
		print("Class Probabilities have been calculated.")
		self.classData = {}
		for i in range(self.class_count):
			_class = self.train_data[self.train_data[self.labelName] == i]
			_class = _class.iloc[:, :-1]
			self.classData[i] = {}
			for attribute in _class:
				self.classData[i][attribute] = {"mean" : np.mean(_class[attribute]), "std" : np.std(_class[attribute])}
	
	def predict(self, inputVector):
		classProb = {}
		ans = None
		maxP = 0
		for i in range(self.class_count):
			classProb[i] = 1
			for attribute in self.train_data.columns:
				try:
					 classProb[i] *= self.calcProb(inputVector[attribute], self.classData[i][attribute]["mean"], self.classData[i][attribute]["std"])
				except Exception as e:
					pass
			if classProb[i] > maxP:
				maxP = classProb[i]
				ans = i
		return ans

	def test(self):
		self.separateClasses()
		correct = 0
		for row,label in zip(self.test_data.iterrows(), self.test_labels):
			if self.predict(row) == label:
				correct += 1
		print("Accuracy is {} %".format(100*correct/len(self.test_data)))

def main():
	df = pd.read_csv("diabetes.csv")
	nb = NaiveBayes(df, "Outcome", 2)
	nb.test()

if __name__ == '__main__':
	main()

# OUTPUT
# Train text split is 0.6
# Class Probabilities have been calculated.
# Accuracy is 69.8051948051948 %