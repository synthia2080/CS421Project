# CS421Project 
Synthia Sasulski - lsasu2

[link to github](https://github.com/synthia2080/CS421Project.git)

## How to Run
You can either run a single essay or a path to a folder of essays

<br><ins>For single essays:</ins><br>
python run_project.py --single_essay "path to essay"

<br><ins>For a folder of essays:</ins><br>
python run_project.py --folder_path "path to folder"

<br>And that's all! The scores will then simply be printed in the terminal.

<br>

## Packages Used
- Numpy
- nltk
- SpaCY
- pandas
- argparse
- os

## Functions/Explanations for scoring

### a-score (num_sentences(), sample_code_1.py)
There were 3 steps in determining where sentences start/end.
1. use nltk sentence tokenizer as a base since its relatively accurate and a good starting off point
2. Get "sub-sentences" based off capitalization. This is done through going through the words and respective POS tags, and checking
if the word is capitlaized. If the word is, and its not a proper noun or part of a manually created list of POS tags that usually can't be at the end of a sentence, we can then assume its a new sentence.
3. Get "sub-sentences" based off POS tag exploitation. For all of the newly created and previous sentences, check for finite verbs. If there are more than one and there are more finite verbs than subordinate/coodinate clause tags, it should indicate multiple sub-sentences, otherwise if theres only 1 finite verb, we're done since its most likely just one sentence. If the first is true, then we look where the finite verbs and subordinate/coordinate clauses are. Then, if one of the tokens dependency is marked as "mark", it indicates a subordinate/coordinate clause so simply append the sentence, otherwise we go until a finite verb is found, continue until a token is found after that can possbily mark the end of  of sentence, if it can then we found another sub-sentence. Simply continue for the sentence until we passed the final finite verb.

The scoring was calculated through simply getting the average number of sentences for high/low essays and using interpolation to get a score from 0-5 based on the manually found averages.

### b-score (spelling_mistakes(), sample_code_1.py)
There are 3 steps to look for spelling mistakes.
1. We utilize a spell checking library: pyspellchecker
2. Then we take the input text and then tokenize the text into individual words using word_tokenize function which is imported from the nltk library.
3. We utilize unknown function from the SpellChecker library to see if there are any spelling mistakes in the words array created from the previous step. Finally, we return the length of the potentially misspelled words in the given text.

### ci-score (agreement(), sample_code_2.py)
There are a few checks to look for subject-verb agreement.
After passing in the newly tokenized sentences from num_sentences(), we check through the POS tags again. First, we iterate through each tokenized sentence to identify the subject using POS tags to determine whether the subject is singular of plural. Then for each sentence, we consider two major verb groups: is the verb auxiliary or not. If the verb is not auxiliary: it is not in the auxiliary verbs list (we have only considered the verb has and do since this are the most common ones, but we can add other common ones), then we ensure that its form matches the subject. There are two cases, for singular subjects ("he", "she", "it"), the verb should not be in plural form. For singular subjects ("I", "you") and plural subjects ("we", "they"), the verb should not be in singular form. This is determined using POS tags for verbs. If there is an error, the error counter increments by 1, and finally the function returns the total number of errors.

### cii-score (verbMistakes(), sample_code_2.py)
There are a few checks to look for verb mistakes.
After passing in the newly tokenized sentences from num_sentences(), we check through the POS tags again. First, we check for a correct verb tense following infinitive. Then we check for missing auxilary verbs through looking at the word's children's dependency. After all that, and counting the number of finite verbs, we check to see if theres only 1 finite verb (from the previous missing verb checks we also checked for the root verb). Once these checks are done, we check for discrepencies between root verb tenses in consecutive sentences.

The scoring was done similarly to the scoring in num_sentences(), however, the number of mistakes as well as the number of verb tense changes between sentences are normalized. This is to account for longer essays possibly having more mistakes simply because theres more room for error. Normalizing the verb tense changes also helps account for correct tense changes between sentences, which was proven difficult to determine reliably. 


