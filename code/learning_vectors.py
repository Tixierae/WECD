import gensim
path_root = '' # TOFILL

with open(path_root + 'full_text_sentences_new.txt', 'r') as my_file:
    sentences = my_file.read().splitlines()
    
sentences_splitted = [elt.split() for elt in sentences]

# training takes a few minutes
model = gensim.models.Word2Vec(sentences_splitted,
                               window = 5, # default 
                               size=300, # dimensionality of the word vectors
                               workers=4, # number of cores to use
                               iter=15, # more is better
                               sample=1e-3, # default, subsampling threhsold
                               negative=5, # default, number of negative examples for each positive one
                               sg=1) # skip-gram

# when training is done
model.init_sims(replace=True)
model.save(path_root + '\\models\\' + 'name_of_model')
