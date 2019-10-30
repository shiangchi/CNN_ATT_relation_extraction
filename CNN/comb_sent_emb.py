

def comb_emb(train_emb, test_emb, output_name) :
	import cPickle as pkl
	import pickle
	
	f_train = open(train_emb, 'rb')
	f_test = open(test_emb, 'rb')
	train_embeddings = pkl.load(f_train)
	test_embeddings = pkl.load(f_test)

	f_train.close()
	f_test.close()

	def merge_dicts(*dict_args):
	    """
	    Given any number of dicts, shallow copy and merge into a new dict,
	    precedence goes to key value pairs in latter dicts.
	    """
	    result = {}
	    for dictionary in dict_args:
	        result.update(dictionary)
	    return result

	z = merge_dicts(train_embeddings, test_embeddings)
	output = open('../corpus/sent_pkl/'+output_name+'.pkl', 'wb')
	
	pickle.dump(z, output)
	output.close()