import time
import math
import operator
import gensim
import argparse
import shelve
import pandas as pd
from gensim.models.word2vec import Word2Vec
from multiprocessing import Pool
from collections import Counter
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer

# define command line arguments
# in that order: path_root, n_jobs, custom_or_google_or_bow, keywords_or_full_text, name_persist

# example command line: python test.py F:\\ 8 custom keywords custom_keywords_10_11_16

parser = argparse.ArgumentParser()
parser.add_argument('path_root', type=str, help='path to required files')
parser.add_argument('n_jobs', type=int, help='number of cores to use')
parser.add_argument('custom_or_google_or_bow', type=str, help='whether to use the custom word vectors or the ones trained on Google News or a bow approach')
parser.add_argument('keywords_or_full_text', type=str, help='whether to compute WMD based on full narrative or associated keywords')
parser.add_argument('name_persist', type=str, help='name under which to persist results')
args = parser.parse_args()

# convert command line arguments
n_jobs = args.n_jobs
path_root = args.path_root
custom_or_google_or_bow = args.custom_or_google_or_bow
keywords_or_full_text = args.keywords_or_full_text
name_persist = args.name_persist

# function that predicts the class of an element
# using a nearest neighbor approach
def process_element(element, my_model_input, text_outcome_training_input, my_index_input, wmd_or_euclidean_input, vect_input, bigram_transformer_input):

	model_inner = my_model_input

	# compute WMD or euclidean distance between every element of the test set/train test
	
	if wmd_or_euclidean_input == 'wmd':
		all_distances = []
		for other_element in text_outcome_training_input:
			#all_distances.append(model_inner.wmdistance(bigram_transformer_input[element[my_index_input].lower().split()], bigram_transformer_input[other_element[my_index_input].lower().split()]))
			all_distances.append(model_inner.wmdistance(element[my_index_input].lower().split(), other_element[my_index_input].lower().split()))

	elif wmd_or_euclidean_input == 'euclidean':
		other_elements = [other_element[my_index_input] for other_element in text_outcome_training_input]
		other_elements_bow = vect_input.transform(other_elements).toarray()
		# compute distances from element to other_element all at once
		element_bow = vect_input.transform([element[my_index_input]]).toarray()
		all_distances = euclidean_distances(other_elements_bow, element_bow)
		# convert into list
		all_distances = [item for sublist in all_distances.tolist() for item in sublist]

	# get index of narratives sorted by distance    
	# by increasing values (default)
	index_sorted = sorted(range(len(all_distances)), key=lambda k: all_distances[k])

	# 5, 10, 15, 20, 25
	predictions_per_neighbor_value = []
	performance_per_neighbor_value = []

	for n_neighbors in [5,10,15,20,25]:

		neighbors = [text_outcome_training_input[i] for i in index_sorted[:n_neighbors]]  

		# 'severity', 'type', 'trade'
		categories = []
		for p in [2,3,4]:
			categories.append([neighbor[p] for neighbor in neighbors])

		# 'severity', 'type', 'trade'
		predictions = []
		for cat in categories:
			my_dict = dict(Counter(cat))

			# number of classes corresponding to max value
			n_max = len([value for value in my_dict.values() if value==max(my_dict.values())])

			# break ties by removing the furthest neighbor
			if (n_max>1):
				cat = cat[:(len(cat)-1)]
				my_dict = dict(Counter(cat))

			# generate predictions as the class corresponding to max value
			predictions.append(max(my_dict.iteritems(), key=operator.itemgetter(1))[0])

		predictions_per_neighbor_value.append(predictions)

		# compute scores
		boolean = []
		for pp, prediction in enumerate(predictions):
		    boolean.append(prediction == element[2+pp])

		performance_per_neighbor_value.append(boolean)

	return([predictions_per_neighbor_value,performance_per_neighbor_value])

def main():
	if custom_or_google_or_bow == 'custom':
		my_model = gensim.models.word2vec.Word2Vec.load(path_root + '\\models\\' + 'skip_gram_9_23_16')
		bigram_transformer = 'foo'
		wmd_or_euclidean = 'wmd'
		vect = 'foo'
		print 'custom model loaded'
		
	elif custom_or_google_or_bow == 'google':
		# load our custom model
		my_model = gensim.models.word2vec.Word2Vec.load(path_root + '\\models\\' + 'skip_gram_9_23_16')
		# replace our custom word vectors with GoogleNews ones (this avoids loading the entire Google vectors which uses a LOT of RAM - approx. 5.5GB)
		my_model.intersect_word2vec_format(path_root + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
		bigram_transformer = 'foo'
		wmd_or_euclidean = 'wmd'
		vect = 'foo'
		print 'google model loaded'
		
	elif custom_or_google_or_bow == 'bow':
	    # load custom stopwords
		with open(path_root + 'custom_stpwds.txt', 'r+') as txtfile:
			custom_stopwords = txtfile.read().splitlines()

		# load words (names) in the vocabulary (same as the skip-gram model)
		with open(path_root + 'word_vectors' + '\\unique_names.txt') as txtfile:
			rownames = txtfile.read().splitlines()
		
		# create bow vectorizer
		vect = CountVectorizer(stop_words=custom_stopwords, vocabulary=rownames)		
		wmd_or_euclidean = 'euclidean'
		my_model = 'foo'
		bigram_transformer = 'foo'
		print 'bag-of-words pipeline created'

    # whether compressed representations of narratives should be used		
	if keywords_or_full_text == 'keywords':
		my_index = 1
		print 'using keyword representation'
	elif keywords_or_full_text == 'full_text':
		my_index = 0
		print 'using full text'
		
	# load classification data set
	full_set = pd.read_csv(path_root + 'classification_data_set.csv', sep=',')

	out_severity = full_set['severity'].tolist()
	out_injury_type = full_set['injury_type'].tolist()
	out_trade = full_set['trade'].tolist()

	# read index of elements to keep
	with open(path_root + 'index_overlap.txt', 'r+') as txtfile:
		index_keep_overlap = txtfile.read().splitlines()
	# convert to integers
	index_keep_overlap = [int(elt) for elt in index_keep_overlap]

	# read narratives
	with open(path_root + 'narratives_full.txt', 'r+') as txtfile:
		narratives = txtfile.read().splitlines()
			
	# read keywords
	with open(path_root + 'keywords_full_new_new.txt', 'r+') as txtfile:
		keywords = txtfile.read().splitlines() 

	print 'data set, index, narratives and keywords loaded'
		
	# map narratives and keywords to outcomes
	text_outcome = []
	for i, k in enumerate(index_keep_overlap):
		text_outcome.append([
		narratives[k],
		keywords[i].strip(),
		out_severity[k],
		out_injury_type[k],
		out_trade[k]
		])

	print 'mapping done'

	# prepare the folds for 4-fold cross-validation
	fold_size = len(text_outcome)/4

	index_folds = []
	index_fold = []
	k = 0

	for i in range(len(text_outcome)):
		index_fold.append(i)
		k += 1
		if k == fold_size:
			print k
			index_folds.append(index_fold)
			index_fold = []
			k = 0
			
	print 'folds created'

	# at every iteration we use 3 folds for training and 1 for testing
	fold_results = []
	t = time.time()
	
	for cv_step in range(4):
		
		training_fold_indexes = [element for element in range(4) if element != cv_step]
		training_indexes = [index_folds[element] for element in training_fold_indexes]
		# flatten list of lists into a list
		training_indexes = [item for sublist in training_indexes for item in sublist]
		
		# here no flattening needed since we only select a single list
		testing_indexes = index_folds[cv_step]

		text_outcome_training = [text_outcome[i] for i in training_indexes]
		text_outcome_testing = [text_outcome[i] for i in testing_indexes]
		
		func = partial(process_element, my_model_input=my_model, text_outcome_training_input=text_outcome_training, my_index_input=my_index, wmd_or_euclidean_input=wmd_or_euclidean, vect_input=vect, bigram_transformer_input = bigram_transformer)
		
		pool = Pool(processes=n_jobs)
		current_results = pool.map(func, text_outcome_testing)
		pool.close()
		pool.join()
		fold_results.append(current_results)
		
		print 'fold #' + str(cv_step + 1) + ' done in ' + str(math.ceil(time.time() - t)) + ' second(s)'
		t = time.time()
    
	# sanity checks
	# length = number of folds
	print len(fold_results)
	# length = number of elements in test set
	print len(fold_results[0])
	# [0]: predictions, [1]: whether they are correct or not 
	print len(fold_results[0][0])
	# number of unique neighbor values tried
	print len(fold_results[0][0][0])
	# severity, type, trade 
	print len(fold_results[0][0][0][0])
	
	print fold_results[0][0][0][0]
	print fold_results[0][0][1][0]
	
	# persist results to disk
	d = shelve.open(path_root + '\\results\\' + name_persist)
	d['fold_results'] = fold_results
	d.close()
	print 'results saved to disk'
	
if __name__ == "__main__":
	main()