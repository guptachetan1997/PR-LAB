from collections import Counter

def get_prior(labels):
	c = Counter(labels)
	sports = c[1]/len(labels)
	return sports, 1-sports

def getProbability(data, labels):
	yes = {}
	no = {}
	words = []
	for idx, item in enumerate(data):
		for word in item.split(" "):
			words.append(word)
			if labels[idx] == 1.0:
				try:
					yes[word] += 1
				except:
					yes[word] = 1
			else:
				try:
					no[word] += 1
				except:
					no[word] = 1
	total = len(set(words))
	for key in yes:
		yes[key] /= total
	for key in no:
		no[key] /= total
	return yes, no

def predict(test, data, labels):
	sports_words, non_sports_words = getProbability(data, labels)
	sports, non_sports = get_prior(labels)
	test_sports = sports
	test_non_sports = non_sports
	for word in test.split(" "):
		try:
			test_sports *= sports_words[word]
		except:
			pass
		try:
			test_non_sports *= non_sports_words[word]
		except:
			pass
	print("Sports : {}, NonSports : {}".format(test_sports, test_non_sports))

def main():
	test = 'it was a good game'
	data = ['a great game','the election was over','very clean match',
	'a clean but forgettable game','it was a close election']
	labels = [1.0,0.0,1.0,1.0,0.0]
	predict(test, data, labels)

if __name__ == '__main__':
	main()

# OUTPUT
# Sports : 0.012244897959183671, NonSports : 0.0002915451895043731