import numpy as np

class Kmeans():

	def __init__(self, X, k, max_iter=100):
		self.X = X
		self.k = k
		self.max_iter = max_iter
	
	def randomCentroids(self):
		self.centroids = self.X[np.random.choice(len(self.X), self.k), :]

	def assignCluster(self):
		self.assigned = [np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in self.centroids]) for x_i in self.X]

	def updateCluster(self):
		self.centroids = [self.X[[thing == k for thing in self.assigned]].mean(axis=0) for k in range(self.k)]
	
	def fit(self):
		self.randomCentroids()
		print(self.centroids)
		for i in range(self.max_iter):
			print("Iteration {}".format(i))
			self.assignCluster()
			self.updateCluster()

def main():
	m1, cov1 = [9, 8], [[1.5, 2], [1, 2]]
	m2, cov2 = [5, 13], [[2.5, -1.5], [-1.5, 1.5]]
	m3, cov3 = [3, 7], [[0.25, 0.5], [-0.1, 0.5]]
	data1 = np.random.multivariate_normal(m1, cov1, 250)
	data2 = np.random.multivariate_normal(m2, cov2, 180)
	data3 = np.random.multivariate_normal(m3, cov3, 100)
	X = np.vstack((data1,np.vstack((data2,data3))))
	np.random.shuffle(X)
	km = Kmeans(X, 3)
	km.fit()
	
if __name__ == '__main__':
	main()