import time
import pickle
import os
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('/home/hsu/Documents/simple_cnn_cdr/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin', binary=True)
unigram = 'induced'
find_word = '-induced'

sim_dict = dict()
def find_similar(find_word, unigram) :

	try :

		if find_word not in sim_dict.keys() :
			sim_dict[find_word]= list()
			similar_list = word_vectors.most_similar(positive=[find_word])
			sim_dict[find_word] = similar_list
	
		else :
			similar_list = sim_dict[find_word]	
		
		for word, score in similar_list :
			if word == unigram :
				return find_word +' '+ unigram
			else :
				return "null"

	except :
		print "?"
		return "null"

similar_word = find_similar("recording", "administration")
print similar_word
