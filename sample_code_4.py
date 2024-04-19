import numpy as np
import pickle as pkl

EMBEDDING_FILE = "w2v.pkl"

def load_w2v(filepath):
    """
        load_w2v function from hw3

        filepath: path of w2v.pkl
        Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
    """
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)
    

def w2v(word2vec, token):
    """
        w2v function from hw3

        word2vec: The pretrained Word2Vec representations as dictionary
        token: A string containing a single token
        Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
    """
    word_vector = np.zeros(300, )

    # [Write your code here:]
    if token in word2vec:
        return word2vec[token]
    
    return word_vector

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
        cosine_similarity function from hw3

        a: A numpy vector of size (x, )
        b: A numpy vector of size (x, )
        Returns: sim (float)
        Where, sim (float) is the cosine similarity between vectors a and b. x is the size of the numpy vector. Assume that both vectors are of the same size.
    """
    aDOTb = np.dot(a,b)

    aMag = np.linalg.norm(a)
    bMag = np.linalg.norm(b)

    if aMag == 0.0 or bMag == 0.0:
        return 0.0

    return aDOTb / (aMag * bMag)

def semanticsPragmatics():

    pass

