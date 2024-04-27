import nltk
from nltk.parse.corenlp import CoreNLPParser

def syntacticWellFormedness(sentence):
    parser = CoreNLPParser(url='http://corenlp.run', tagtype='pos')
    dependency_tree = next(parser.raw_parse(sentence))
    mistakes = 0
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
                            break
                        else:
                            mistakes += 1

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

    return mistakes


example_sentence1 = "My dog with a broken leg I not want"
example_sentence2 = "I do not want my dog with a broken leg"
example_sentence3 = "Because I think the science and technology are developing, I want to pursue those fields"
example_sentence4 = "I came when he being sick"

mistake_count1 = syntacticWellFormedness(example_sentence1)
mistake_count2 = syntacticWellFormedness(example_sentence2)
mistake_count3 = syntacticWellFormedness(example_sentence3)
mistake_count4 = syntacticWellFormedness(example_sentence4)

print("Number of mistakes found in sentence 1:", mistake_count1)
print("Number of mistakes found in sentence 2:", mistake_count2)
print("Number of mistakes found in sentence 3:", mistake_count3)
print("Number of mistakes found in sentence 4:", mistake_count4)
