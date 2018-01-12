import numpy as np

class PCA():
	
	def __init__(self, data, dimensions):
		self.data = np.array(data)
		self.dimensions = dimensions

	def get_cov_matrix(self):
		zero_mean = (self.data - np.mean(self.data, axis=0))
		self.cov = (1/(self.data.shape[0]-1)) * zero_mean.T.dot(zero_mean)
		print("The covariance matrix is : ")
		print(self.cov)
	
	def evd(self):
		eigenValues, eigenVectors = np.linalg.eig(self.cov)
		print("The EigenValues are : ")
		print(eigenValues)
		print("The corresponding EigenVectors are : ")
		print(eigenVectors)
		eigen_dict = [{"eigenvalue" : x, "eigenvector" : y} for x,y in zip(eigenValues, eigenVectors)]
		self.eigen_dict = sorted(eigen_dict, key= lambda k : -k["eigenvalue"])
		self.total_var = np.sum(eigenValues)
	
	def reduce_dim(self):
		print("Original Dimensions : {}".format(self.data.shape))
		self.get_cov_matrix()
		self.evd()
		self.final_var = np.sum([self.eigen_dict[i]["eigenvalue"] for i in range(self.dimensions)])
		self.reduced_data = "p"
		Q = np.array([self.eigen_dict[i]["eigenvector"] for i in range(self.dimensions)])
		self.reduced_data = self.data.dot(Q.T)
		print("Variance Retained : {}%".format(self.final_var*100.0/self.total_var))
		print("Final Dimensions : {}".format(self.reduced_data.shape))

def main():
	x = np.array([
			[2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
			[2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]]	)
	x = x.T
	pca = PCA(x, 1)
	pca.reduce_dim()

if __name__ == '__main__':
	main()

# OUTPUT : 
# Original Dimensions : (10, 2)
# The covariance matrix is : 
# [[ 0.61655556  0.61544444]
#  [ 0.61544444  0.71655556]]
# The EigenValues are : 
# [ 0.0490834   1.28402771]
# The corresponding EigenVectors are : 
# [[-0.73517866 -0.6778734 ]
#  [ 0.6778734  -0.73517866]]
# Variance Retained : 96.31813143486458%
# Final Dimensions : (10, 1)