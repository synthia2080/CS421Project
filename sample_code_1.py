import numpy as np
import nltk
import pandas as pd
from nltk import tokenize, word_tokenize
from spellchecker import SpellChecker
import os
import spacy

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
    low_sentence_num = 0
    low_sentence_totalSentenceNum = 0
    high_sentence_num = 0
    high_sentence_totalSentenceNum = 0
    for name, essay in essays:
        numSentence, new_sentences = num_sentences(essay)

        df = index[index['filename'] == name]
        score = df['grade'].iloc[0]

        if score == "high":
            high_sentence_num += 1
            high_sentence_totalSentenceNum += numSentence
        else:
            low_sentence_num += 1
            low_sentence_totalSentenceNum += numSentence

    #compute averages
    high_average = high_sentence_totalSentenceNum / high_sentence_num
    low_average = low_sentence_totalSentenceNum / low_sentence_num

    print(f"Average number of sentences in 'high' grades: {high_average}")
    print(f"Average number of sentences in 'low' grades: {low_average}")



def num_sentences(txt):
    """
    Tokenizes essays by sentence, then for each sentence, get POS tags and search for any extra sentences missed by
    the nltk sentence tokenizer

    txt: essay as a string
    returns: score as an int
    """
    #Get base number of sentences through sentence tokenizer
    tokenized_sentences = tokenize.sent_tokenize(txt)
    processor = spacy.load("en_core_web_sm")
            
    comma_list = [',', ';']
    not_EOS_pos_tags = ["DT","IN", "CC","DT", "PRP$", "WP$", "IN", "VB", "VBP", "VBZ", "VBD"] #Tags that shouldn't be at end of sentence
    start_tags = ["PRP", "DT", "WP", "IN", "RB", "NN", "NNP"] #Possible tags at start of sentence
    new_sentences = [] #holds all new sentences minus the last one

    # Checking for missed sentences based off of capitalization first
    for s in tokenized_sentences:
        #get POS tags for analyzing
        tokenized_words = word_tokenize(s)
        tokenized_words = ["I" if word == "i" else word for word in tokenized_words] #Helps POS tagger appropriately label "I"
        POS_tags = nltk.pos_tag(tokenized_words)

        #holds the new sentence
        holder_sentence = ""

        #Go through each word & its POS tag
        for i2, (word, tag) in enumerate(POS_tags):
            if i2 == 0:
                holder_sentence += f"{word} "
                continue
            
            word_before = POS_tags[i2-1][0]
            tag_before = POS_tags[i2-1][1]

            #Assume capitalized, non-proper nouns is a marker for new sentence
            if word[0].isupper() and tag != 'NNP' and word_before not in comma_list and tag_before not in not_EOS_pos_tags:
                new_sentences.append(holder_sentence)
                holder_sentence = ""

            holder_sentence += f"{word} "
        new_sentences.append(holder_sentence)


    notfiniteVerbTags = ['VB', 'VBG', 'VBN'] #Verb tags not included here are finite
    sub_coord_tags = ['CC', 'IN', 'WP', 'WDT', 'WP$', 'RB'] #Tags usually indicating subordinate/coordinate clauses
    start_tags = ["PRP", "DT", "WP", "IN", "RB", "NN", "NNP"] #Tags that usually are in beginning of sentences
    not_end_pos = ['PART', 'DET' , 'AUX', 'INTJ', 'ADP', 'CONJ', 'CCONJ', 'SCONJ'] #.pos_ tags that shouldn't be at end of sentence
    repeatable_conj = ['and', 'or', 'nor'] #conjucations that can be repeated multiple times in sentence


    new_new_sentences = [] #Final list of sentences
    for s in new_sentences:
        POS_tags = processor(s)

        finite_verbs = []
        this_sub_coord_tags = []

        #Get number of finite verbs and get each sub/coord clause tags 
        for value in POS_tags:
            if value.tag_ not in notfiniteVerbTags and value.tag_[0] == "V":
                finite_verbs.append(value)
            elif value.tag_ in sub_coord_tags:
                this_sub_coord_tags.append(value)

        # # For debugging
        # for token in POS_tags:
        #     print(f"{token.text} {token.tag_} {token.pos_} {token.dep_}")

        #Sentences with less than 1 finite verbs should be complete
        if len(finite_verbs) <= 1:
            new_new_sentences.append(s)
            continue
        

        #If There are more finite verbs than coord/sub clause indicating words, there must be a hidden sentence, otherwise sentence should be correct
        if len(finite_verbs) > len(this_sub_coord_tags):
            holder_sentence = ""
            temp_fin_count = 0
            on_finite = False
            for i, token in enumerate(POS_tags):
                holder_sentence += f"{token.text} "

                #append sentence if there is subordinate/coordinate clause
                if token.dep_ == "mark":
                    new_new_sentences.append(s)
                    holder_sentence = ""
                    break
    
                #Check if current word is finite verb
                if token.tag_ not in notfiniteVerbTags and token.tag_[0] == "V":
                    temp_fin_count += 1
                    on_finite = True
                
                #If we are past a finite verb, check if end of sentence is possible (whether word is able to be at the end)
                if i != len(POS_tags)-1 and on_finite == True and temp_fin_count < len(finite_verbs) and POS_tags[i+1].tag_ in start_tags and POS_tags[i+1].pos_ not in not_end_pos and POS_tags[i+1].tag_ not in this_sub_coord_tags and token.text.lower() not in repeatable_conj:
                    new_new_sentences.append(holder_sentence)
                    holder_sentence = ""
                    on_finite = False
                    continue

            #Append remaining sentence in string
            if holder_sentence != "":
                new_new_sentences.append(holder_sentence)
        else:
            new_new_sentences.append(s)

    num_sentences = len(new_new_sentences)

    #Calculate score 
    if num_sentences <= 0:
        return 0, new_new_sentences
    elif num_sentences >= 22:
        return 5, new_new_sentences
    else:
        return (num_sentences**2 / 22.0**2) * 5, new_new_sentences



def spelling_mistakes(txt):
    spelling = SpellChecker()
    words_array = word_tokenize(txt)
    commonly_mispelled_words = spelling.unknown(words)
    return len(commonly_mispelled_words)

def main():

    #Simply here for testing
    test = "Most people do not walk to work; instead, they drive or take the train. I love trains and I love brains and I love hating cars I am cool therefore you are cool too. I want to do well I am sad i am happy. i am cool i am not cool he is dumb. After he and I finished my homework, I went to bed and also brushed my teeth."
    # test = "After he and I finished my homework, I went to bed and also brushed my teeth."
    # test = "I remember that I went to see the eclipse"
    # test = "I want to do well I am sad i am happy."
    # test = "i am not cool he is dumb learning is the best thing ever."
    # test = 'I remember that I went to see the eclipse'
    print(num_sentences(test))

    getAverageSentCount()
    print(spelling_mistakes(test))
    #Holding here in case its needed for future use
    # "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


if __name__ == "__main__":
    main()
