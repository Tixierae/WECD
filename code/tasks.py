import numpy as np

import matplotlib.pyplot as plt

from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# ==============================

def does_not_match_by_hand(my_names):
    similarities_to_all_others = []
    for my_name in my_names:
        index_avoid = my_names.index(my_name)
        my_names_others = [my_names[i] for i in range(len(my_names)) if i!=index_avoid]
        
        distances_inner = []
        for name_inner in my_names_others:
            distances_inner.append(model.similarity(my_name,name_inner))
        
        # store the average
        similarities_to_all_others.append(sum(distances_inner)/float(len(distances_inner)))   
        
    return similarities_to_all_others

# function that returns word vector as numpy array
def my_vector_getter(word, my_coordinates):
    index = words.index(word)
    word_array = my_coordinates[index].ravel()
    return (word_array)

def plot_concept(names, pc_x, pc_y):
    my_vectors = []
    for name in names:
        my_vectors.append(my_vector_getter(name, new_coordinates))
        
    dim_1_coords = [element[pc_x] for element in my_vectors]
    dim_2_coords = [element[pc_y] for element in my_vectors]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dim_1_coords, dim_2_coords, 'ro')
 
    for x, y, name in zip(dim_1_coords , dim_2_coords, names):                                                
        ax.annotate(name, xy=(x, y))
 
    plt.grid()
    plt.show()

# ==============================

path_root = '' # TOFILL

model = Word2Vec.load(path_root + '\\models\\' + 'skip_gram_9_23_16') 

words = model.vocab.keys()

# store custom word vectors as list of lists
word_vectors = []
i = 0
for elt in words:
    word_vectors.append(model[elt].tolist())
    i += 1
    if i % round(len(model.vocab)/10) == 0:
        print i

# ==============================
		
### most similar ###

# based on cosine similarity
  
model.most_similar('acid')
model.most_similar('2x4')
model.most_similar('foreman')
model.most_similar('autodesk')

### does not match ###

does_not_match_by_hand(['pipe', 'roof', 'trench', 'cables' ,'ground'])    
model.doesnt_match("pipe roof trench cables ground".split()) # the lowest value is the vector than does not match

does_not_match_by_hand("asbestos silica rebar fiberglass dust".split())  
model.doesnt_match("asbestos silica rebar fiberglass dust".split())

model.doesnt_match("carpenter employee laborer electrician bim".split())
does_not_match_by_hand("carpenter employee laborer electrician bim".split())  

model.doesnt_match("car truck drill excavator manlift crane".split())
does_not_match_by_hand("car truck drill excavator manlift crane".split())

model.doesnt_match("hernia injury fracture burn building sprain".split())

### A is to B as C is to ... ###

# read 'word in 3rd position' is to 'word in 2nd position'  as 'word in 1st position' is to response
model.most_similar(positive=['concrete', 'ironworker'], negative=['beams'])
model.most_similar(positive=['wood', 'mason'], negative=['brick'])
model.most_similar(positive=['bolts', 'hammer'], negative=['nails'])
model.most_similar(positive=['autodesk', 'truck'], negative=['volvo'])

### plots ###

# note: every time they are trained, word vectors change (random initialization, tochastic nature of gradient descent... -> weights stabilize in a different state each time). Overall, the same regularities should be encoded every time, but they are not always captured by the same dimensions...

# project words into a lower-dimensional space using PCA
# PCA itself is stochastic due to numerical approximations
my_pca = PCA(n_components=10)
new_coordinates = my_pca.fit_transform(np.array(word_vectors))
new_coordinates.shape

plot_concept(['ankle','sprain','head','concussion','irritation','skin'],0,1)

plot_concept(['driving','truck','vibrating','concrete','welding','metal','wearing','ppe'],0,1)

plot_concept(['bricks','mason','electrician','wire','welder','rod'],0,1)

### WMD ###
 
sentence_1 = 'worker was using hammer to drive nails'.lower().split()
sentence_2 = 'employee was driving nails with hammer'.lower().split()
sentence_3 = 'the president addressed the press in chicago'.lower().split()


# remove stopwords
sentence_1 = [w for w in sentence_1 if w not in stopwords]
sentence_2 = [w for w in sentence_2 if w not in stopwords]
sentence_3 = [w for w in sentence_3 if w not in stopwords]


# compute WMD
print model.wmdistance(sentence_1, sentence_2)
print model.wmdistance(sentence_1, sentence_3)
