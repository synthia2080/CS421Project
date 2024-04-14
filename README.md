# CS421Project

## How to Run
You can either run a single essay or a path to a folder of essays

<br><ins>For single essays:</ins><br>
python run_project.py --single_essay "path to essay"

<br><ins>For a folder of essays:</ins><br>
python run_project.py --folder_path "path to folder"

<br>And that's all! The scores will then simply be printed in the terminal.

<br>

## Explanation/Process for scoring

### a-score (num_sentences)
There were 3 steps in determining where sentences start/end.
1. use nltk sentence tokenizer as a base since its relatively accurate and a good starting off point
2. Get "sub-sentences" based off capitalization. This is done through going through the words and respective POS tags, and checking
if the word is capitlaized. If the word is, and its not a proper noun or part of a manually created list of POS tags that usually can't be at the end of a sentence, we can then assume its a new sentence.
3. Get "sub-sentences" based off POS tag exploitation. For all of the newly created and previous sentences, check for finite verbs. If there are more than one and there are more finite verbs than subordinate/coodinate clause tags, it should indicate multiple sub-sentences, otherwise if theres only 1 finite verb, we're done since its most likely just one sentence. If the first is true, then we look where the finite verbs and subordinate/coordinate clauses are. Then, if one of the tokens dependency is marked as "mark", it indicates a subordinate/coordinate clause so simply append the sentence, otherwise we go until a finite verb is found, continue until a token is found after that can possbily mark the end of  of sentence, if it can then we found another sub-sentence. Simply continue for the sentence until we passed the final finite verb.

The scoring was calculated through simply getting the average number of sentences for high/low essays and using interpolation to get a score from 0-5 based on the manually found averages.

### b-score (spelling mistakes)
explanation...

### ci-score (verb agreement)
explanation...

### cii-score (verb mistakes)
There are a few checks to look for verb mistakes.
Afte passing in the newly tokenized sentences from num_sentences(), we check through the POS tags again. First, we check for a correct verb tense following infinitive. Then we check for missing auxilary verbs through looking at the word's children's dependency. After all that, and counting the number of finite verbs, we check to see if theres only 1 finite verb (from the previous missing verb checks we also checked for the root verb). Once these checks are done, we check for discrepencies between root verb tenses in consecutive sentences.

The scoring was done similarly to the scoring in num_sentences(), however, the number of mistakes was also normalized for the number of sentences so that higher scoring essays don't get penalized as much.


