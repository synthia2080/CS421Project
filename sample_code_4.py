import numpy as np
import pickle as pkl
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os
import pandas as pd
from sample_code_1 import num_sentences
import gensim.downloader as api
import spacy

spacy_processor = spacy.load("en_core_web_sm")

EMBEDDING_FILE = "w2v.pkl"

def load_w2v(filepath):
    """
        load_w2v function from hw3

        filepath: path of w2v.pkl
        Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
    """
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)
    

# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    """
        w2v function from hw3 

        ***May not be needed

        word2vec: The pretrained Word2Vec representations as dictionary
        token: A string containing a single token
        Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
    """
    word_vector = np.zeros(300, )

    # [Write your code here:]
    if token in word2vec:
        return word2vec[token]
    
    return word_vector


# This function embeds the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    """
        string2vec function from hw3 

        word2vec: The pretrained Word2Vec model
        user_input: A string of arbitrary length
        Returns: A 300-dimensional averaged Word2Vec embedding for that string
    """
    #Tokenize words
    tokenized_words = word_tokenize(user_input)  
    POS_tags = nltk.pos_tag(tokenized_words)
    spacy_tags = spacy_processor(user_input)

    content_tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    #might need to check POS tags for content words
    #Get embeddings for each token
    word2vecArray = []
    for token in spacy_tags:
        if token.tag_ in content_tags and not token.is_punct and not token.is_stop and token.text in word2vec:
            word2vecArray.append(word2vec[token.text])

    #Get the averages
    word2vecArray = np.array(word2vecArray)
    if word2vecArray.shape[0] == 0:
        return np.zeros(300,)

    averageArray = word2vecArray.mean(axis=0)

    return averageArray

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


def semanticsPragmatics(prompt, tokenized_sentences):
    word2vec = load_w2v(EMBEDDING_FILE)

    #Calculate average sentence embeddings for each sentence in essay
    essay_sentenceEmbeddings = []
    for s in tokenized_sentences:
        essay_sentenceEmbeddings.append(string2vec(word2vec, s))

    # ***** subscore d.i *****
    prompt_sentenceTokenized = sent_tokenize(prompt)
    topicalPart = prompt_sentenceTokenized[1]

    prompt_avgEmbedding = string2vec(word2vec, topicalPart)

    prompt_essay_embeddingSum = 0

    for s in essay_sentenceEmbeddings:
        prompt_essay_embeddingSum += cosine_similarity(prompt_avgEmbedding, s)

    prompt_essay_embeddingAvg = (prompt_essay_embeddingSum / float(len(essay_sentenceEmbeddings))) * -100
    di_score = 0

    # return prompt_essay_embeddingAvg
    correctPromptHighThreshold = -22.5
    correctPromptLowThreshold = -23
    incorrectPromptHighThreshold = -37.2
    incorrectPromptLowThreshold = -38.8

    if prompt_essay_embeddingAvg >= correctPromptHighThreshold:
        di_score = 5  
    elif correctPromptLowThreshold <= prompt_essay_embeddingAvg < correctPromptHighThreshold:
        di_score = 4
    elif incorrectPromptHighThreshold <= prompt_essay_embeddingAvg:
        di_score = 3
    elif incorrectPromptLowThreshold <= prompt_essay_embeddingAvg < incorrectPromptHighThreshold:
        di_score = 2
    else:
        di_score = 1

    # ***** subscore d.ii *****
    dii_score = 0
    # essayEmbeddingsAvg = np.array(essay_sentenceEmbeddings).mean(axis=1)
    cos_all = []
    for i, cs in enumerate(essay_sentenceEmbeddings):
        if i == len(essay_sentenceEmbeddings)-1:
            break

        s1 = cs
        s2 = essay_sentenceEmbeddings[i+1]

        cosine_sim = cosine_similarity(s1, s2)
        cos_all.append(cosine_sim)

    cos_all = np.array(cos_all)
    sde = np.std(cos_all)
    # print(f"SDE: {sde}")
    sde = sde * 1000

    high_threshold = 145
    low_threshold = 146
    if sde <= high_threshold:
        dii_score = 1
    elif sde >= low_threshold:
        dii_score = 5
    else:
        dii_score = 1 + 4 * (sde - high_threshold) / (low_threshold - high_threshold)

    return di_score, dii_score



# prompt = "Do you agree or disagree with the following statement?		Most advertisements make products seem much better than they really are.		Use specific reasons and examples to support your answer."
# tokenized_sentences = ["sdgsdg", "sdgsdgdg"]
# semanticsPragmatics(prompt, tokenized_sentences)

# word2vec = load_w2v(EMBEDDING_FILE)
# print(string2vec(word2vec, "i love nlp"))

# getAverageSentCount()

# prompt = "Do you agree or disagree with the following statement?		Most advertisements make products seem much better than they really are.		Use specific reasons and examples to support your answer.	"
# essay_path = "1079196.txt"

# prompt = "Do you agree or disagree with the following statement?		Most advertisements make products seem much better than they really are.		Use specific reasons and examples to support your answer.	"
# # # prompt = "Do you agree or disagree with the following statement?		Successful people try new things and take risks rather than only doing what they already know how to do well.		Use reasons and examples to support your answer.	"
# essay_path = "1449555.txt"

# # essay = ""

# with open(essay_path, 'r') as file:
#     essay = file.read()

# _, tokenized_sents = num_sentences(essay)
# prompt_essay_embeddingAvg = semanticsPragmatics(prompt, tokenized_sents)

# # prompt_essay_embeddingAvg = semanticsPragmatics("Testing? Test Prompt. Testing", ["Hello world", "I like the test prompt"])

# print(prompt_essay_embeddingAvg)
