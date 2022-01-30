"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

import numpy as np
import math
from collections import Counter
import operator


def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words)
                test data (list of sentences, no tags on the words)
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''

        posList_word = {}    # dictionary; key:word, value: {tag, count} (dictionary: key: tag, value: count)
        posList_tag = {}        # dictionary; key: tag, vakye: count

        ## Counting ##
        for seq in train:
                for currWord in seq:

                        # if the word is an unseen word
                        if currWord[0] not in posList_word:
                                posList_word[currWord[0]] = {currWord[1]: 1}
                        else:
                                # if the seen word tag is new
                                if currWord[1] not in posList_word[currWord[0]]:
                                        posList_word[currWord[0]][currWord[1]] = 1
                                # if the seen word tag is not new
                                else:
                                        posList_word[currWord[0]][currWord[1]] += 1

                        # if the tag is an unseen tag
                        if currWord[1] not in posList_tag:
                                        posList_tag[currWord[1]] = 1
                        else:
                                posList_tag[currWord[1]] += 1


        ## Createing the output array ##
        test_output = []  # array of arrays of (word, tag)
        currSentence_output = []

        for sentence in test:
        
                for currWord in sentence:
                        # if the word is a seen word
                        if currWord in posList_word:
                                keyMax = max(posList_word[currWord], key = posList_word[currWord].get)
                        # if the word is an unseen word --> guess the tag that's seen the most often in training dataset
                        else:
                                keyMax = max(posList_tag, key = posList_tag.get)

                        currSentence_output.append((currWord, keyMax))

                test_output.append(currSentence_output)
                currSentence_output = []


        return test_output
