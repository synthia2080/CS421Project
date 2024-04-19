import numpy as np
import pickle as pkl
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os
import pandas as pd
from sample_code_1 import num_sentences

EMBEDDING_FILE = "w2v.pkl"

def getAverageSentCount():
    """
    Gets averages of number of sentences for high and low grades
    """
    index = pd.read_csv("essays_dataset/index.csv", delimiter=";")
    pathToEssays = "essays_dataset/essays/"
    essays = []

    #read in essays 
    for filename in os.listdir(pathToEssays):
        file_path = os.path.join(pathToEssays, filename)

        with open(file_path, 'r') as file:
            text = file.read()
            essays.append((filename, text))

    #Count number of sentences for all essays

    sampled_prompts = ["Do you agree or disagree with the following statement?		Most advertisements make products seem much better than they really are.		Use specific reasons and examples to support your answer.	",
                       "Do you agree or disagree with the following statement?		Successful people try new things and take risks rather than only doing what they already know how to do well.		Use reasons and examples to support your answer.	"]

    low_sentence_num = 0
    low_sentence_totalCosineSim = 0
    high_sentence_num = 0
    high_sentence_totalCosineSim= 0
    maxSampledEssays = 5
    essays_used = []
    #Get average for correct prompt essays
    print("Calculating average cosine for correct prompt...")
    for name, essay in essays:
        df = index[index['filename'] == name]
        prompt = df['prompt'].iloc[0]
        if prompt != sampled_prompts[0]:
            continue
        score = df['grade'].iloc[0]
        
        #Check if we already read up to max number of sample for either high or low essays
        if score == "high" and high_sentence_num >= maxSampledEssays:
            continue
        elif score == "low" and low_sentence_num >= maxSampledEssays:
            continue
        
        # print(f" ****{name}")
        _, tokenized_sentences = num_sentences(essay)
        promp_essay_cosineSim = semanticsPragmatics(sampled_prompts[0], tokenized_sentences)
        
        if score == "high":
            high_sentence_num += 1
            high_sentence_totalCosineSim += promp_essay_cosineSim
        else:
            low_sentence_num += 1
            low_sentence_totalCosineSim += promp_essay_cosineSim

        essays_used.append((name, score, essay))

        #Exit if we read up to max for both high and low essays
        if high_sentence_num >= maxSampledEssays and low_sentence_num >= maxSampledEssays:
            break

        print(f"numHighEssays: {high_sentence_num}, numLowEssays: {low_sentence_num}, {name}, {score}")

    print()
    #compute averages
    correctPrompt_high_average = high_sentence_totalCosineSim / high_sentence_num
    correctPrompt_low_average = low_sentence_totalCosineSim / low_sentence_num

    print(f"Average cosine similarity for correct prompt in 'high' grades: {correctPrompt_high_average}")
    print(f"Average cosine similarity for correct prompt in 'low' grades: {correctPrompt_low_average}")
    print()

    #Get averages for incorrect prompt-essay combo
    low_sentence_num = 0
    low_sentence_totalCosineSim = 0
    high_sentence_num = 0
    high_sentence_totalCosineSim= 0
    maxSampledEssays = 5
    #Get average for correct prompt essays
    print("Calculating average cosine for incorrect prompt...")
    for name, score, essay in essays_used:
        #Check if we already read up to max number of sample for either high or low essays
        if score == "high" and high_sentence_num >= maxSampledEssays:
            continue
        elif score == "low" and low_sentence_num >= maxSampledEssays:
            continue

        _, tokenized_sentences = num_sentences(essay)
        promp_essay_cosineSim = semanticsPragmatics(sampled_prompts[1], tokenized_sentences)

        if score == "high":
            high_sentence_num += 1
            high_sentence_totalCosineSim += promp_essay_cosineSim
        else:
            low_sentence_num += 1
            low_sentence_totalCosineSim += promp_essay_cosineSim
        #Exit if we read up to max for both high and low essays
        if high_sentence_num >= maxSampledEssays and low_sentence_num >= maxSampledEssays:
            break
        print(f"numHighEssays: {high_sentence_num}, numLowEssays: {low_sentence_num}, {name}, {score}")

    incorrectPrompt_high_average = high_sentence_totalCosineSim / high_sentence_num
    incorrectPrompt_low_average = low_sentence_totalCosineSim / low_sentence_num

    print(f"Average cosine similarity for incorrect prompt in 'high' grades: {incorrectPrompt_high_average}")
    print(f"Average cosine similarity for incorrect prompt in 'low' grades: {incorrectPrompt_low_average}")

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

    content_tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    #might need to check POS tags for content words
    #Get embeddings for each token
    word2vecArray = []
    for token, tag in POS_tags:
        if tag in content_tags and token in word2vec:
            word2vecArray.append(word2vec[token])

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

    prompt_essay_embeddingAvg = prompt_essay_embeddingSum / float(len(essay_sentenceEmbeddings))

    return prompt_essay_embeddingAvg


    # ***** subscore d.ii *****



# prompt = "Do you agree or disagree with the following statement?		Most advertisements make products seem much better than they really are.		Use specific reasons and examples to support your answer."
# tokenized_sentences = ["sdgsdg", "sdgsdgdg"]
# semanticsPragmatics(prompt, tokenized_sentences)

# word2vec = load_w2v(EMBEDDING_FILE)
# print(string2vec(word2vec, "i love nlp"))

getAverageSentCount()

# prompt = "Do you agree or disagree with the following statement?		Most advertisements make products seem much better than they really are.		Use specific reasons and examples to support your answer.	"
# essay_path = "1079196.txt"

# prompt = "Do you agree or disagree with the following statement?		Most advertisements make products seem much better than they really are.		Use specific reasons and examples to support your answer.	"
# essay_path = "1449555.txt"

# essay = ""
# with open(essay_path, 'r') as file:
#     essay = file.read()

# _, tokenized_sents = num_sentences(essay)
# prompt_essay_embeddingAvg = semanticsPragmatics(prompt, tokenized_sents)

# print(prompt_essay_embeddingAvg)
