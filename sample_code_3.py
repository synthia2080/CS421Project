import nltk
from nltk.parse.corenlp import CoreNLPParser

def syntacticWellFormedness(tokenized_sentences):
    parser = CoreNLPParser(url='http://corenlp.run', tagtype='pos')
    mistakes = 0
    
    for sentence in tokenized_sentences:
        dependency_tree = next(parser.raw_parse(sentence))
        root = dependency_tree.label()

        if root.startswith('VB'):
            mistakes += 1
        elif root in ['SQ', 'SBARQ']:
            if 'aux' not in dependency_tree.leaves() and 'wh' not in dependency_tree.leaves():
                mistakes += 1

        for subtree in dependency_tree.subtrees():
            if isinstance(subtree, nltk.tree.Tree):
                for child in subtree:
                    if isinstance(child, nltk.tree.Tree):
                        if subtree.label() in ['NN', 'NNS', 'NNP', 'NNPS']:
                            if child.label().startswith('DT'):
                                mistakes += 1
                                break

                        elif subtree.label() == 'PP':
                            if child.label() == 'IN':
                                break
                            else:
                                mistakes += 1

                        elif subtree.label() == 'SBAR':
                            if 'VB' in child.leaves() or 'VBG' in child.leaves() or 'mark' in child.leaves():
                                break
                            else:
                                mistakes += 1

        if 'nsubj' not in dependency_tree.leaves():
            mistakes += 1
        
    normalized_mistakes = mistakes / len(tokenized_sentences)
    print(mistakes)
    high_threshold = 0.1450644548321961
    low_threshold = 0.39475772146296434
    if normalized_mistakes < high_threshold:
        return 5
    elif normalized_mistakes > low_threshold:
        return 0
    else:
        return 1 + 3 * (normalized_mistakes - high_threshold) / (low_threshold - high_threshold)


