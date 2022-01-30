# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

from nltk import tokenize  # word_Stemmer
from nltk import stem

# from template.reader import load_dataset   # PorterStemmer()         from nltk.stem.porter 
from reader import load_dataset

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.01, pos_prior=0.8,silently=False):
    """Determine whether the alien touches a wall

        Args:
            train_set:
            train_labels:
            dev_set:
            laplace:        The Lapalace Smoothing parameter alpha
            pos_prior:      The prior probability P(Type=Positive) 
                                Adjust your definition of naiveBayes so that the default value for pos_prior
                                is appropriate for the development dataset.
            silently:

        Return:
            a list containing labels for each of the reviews in the development set 
            (label order should be the same as the document order in the given development set)
    """

    # Counter
    # https://docs.python.org/3/library/collections.html#collections.Counter
    # Dictionary
    # https://docs.python.org/3/library/stdtypes.html#typesmapping
    # https://stackoverflow.com/questions/1024847/how-can-i-add-new-keys-to-a-dictionary
        
  
    ### Training Phase ###
    pos_count = {}      # dictionary -- key: word, value: count     in positive reviews
    neg_count = {}      # dictionary -- key: word, value: count     in negative reviews

    idx_label = 0
    pos_num_words_total = 0
    neg_num_words_total = 0


    for review in train_set:
        for word in review:
            if (train_labels[idx_label] == 1):
                
                if word in pos_count:
                    pos_count[word] = pos_count[word] + 1
                else :
                    pos_count[word] = 1

            else :
                
                if word in neg_count:
                    neg_count[word] = neg_count[word] + 1
                else:
                    neg_count[word] = 1 
        idx_label = idx_label + 1



    neg_prior = 1 - pos_prior
   
    pos_num_words_total = sum(pos_count.values())
    neg_num_words_total = sum(neg_count.values())
    
    
    
    ### Development Phase ### 
    yhats = []
    pos_posterior = 0
    neg_posterior = 0


    len_pos = len(pos_count)
    len_neg = len(neg_count)
    

    for doc in tqdm(dev_set,disable=silently):
        for word in doc:
            if word in pos_count:
                pos_posterior = pos_posterior + math.log((pos_count[word] + laplace) / (pos_num_words_total + laplace * (len_pos + 1)))
            else:
                pos_posterior = pos_posterior + math.log(laplace / (pos_num_words_total + laplace * (len_pos + 1)))


            if word in neg_count:
                neg_posterior = neg_posterior + math.log((neg_count[word] + laplace) / (neg_num_words_total + (laplace * (len_neg + 1))))
            else:
                neg_posterior = neg_posterior + math.log(laplace / (neg_num_words_total + (laplace * (len_neg + 1))))
    
        pos_posterior += math.log(pos_prior)       # in log
        neg_posterior += math.log(neg_prior)       # in log

        
        if (pos_posterior > neg_posterior):
            yhats.append(1)

        else:
            yhats.append(0)

        pos_posterior = 0
        neg_posterior = 0
        
    
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)


    # yhats = []              # prediction result of your bayes model on dev set
    # for doc in tqdm(dev_set,disable=silently):
    #     yhats.append(-1)
    # return yhats
    
    return yhats
   






# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


# main function for the bigrammixture model 999999995 0.0045
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=0.5, silently=False):


    ### Training Phase ###
    uni_pos_count = {}      # dictionary -- key: word, value: count     in positive reviews
    uni_neg_count = {}      # dictionary -- key: word, value: count     in negative reviews
    bi_pos_count = {}
    bi_neg_count = {}

    idx_label = 0
    pos_num_words_total = 0
    neg_num_words_total = 0

    # idx_word = 0
    # temp_idx = 0

    for review in train_set:
        # unigram
        for word in review:
            if (train_labels[idx_label] == 1):
                
                if word in uni_pos_count:
                    uni_pos_count[word] = uni_pos_count[word] + 1
                else :
                    uni_pos_count[word] = 1

            else :
                
                if word in uni_neg_count:
                    uni_neg_count[word] = uni_neg_count[word] + 1
                else:
                    uni_neg_count[word] = 1 

        # bigram
        for idx_word in range(len(review)-1):
            if (train_labels[idx_label] == 1):
            
                # bigram
                if (review[idx_word], review[idx_word+1]) in bi_pos_count:
                    bi_pos_count[(review[idx_word], review[idx_word+1])] = bi_pos_count[(review[idx_word], review[idx_word+1])] + 1
                else:
                    bi_pos_count[(review[idx_word], review[idx_word+1])] = 1

            else:
                # bigram
                if (review[idx_word], review[idx_word+1]) in bi_neg_count:
                    bi_neg_count[(review[idx_word], review[idx_word+1])] = bi_neg_count[(review[idx_word], review[idx_word+1])] + 1
                else:
                    bi_neg_count[(review[idx_word], review[idx_word+1])] = 1
        

        idx_label += 1







    # for review in train_set:
    #     for idx_word in range(len(review)-1):
    #         if (train_labels[idx_label] == 1):
    #             # unigram
    #             if (review[idx_word] in uni_pos_count):
    #                 uni_pos_count[review[idx_word]] += 1
    #             else:
    #                 uni_pos_count[review[idx_word]] = 1


    #             # bigram
    #             if (review[idx_word], review[idx_word+1]) in bi_pos_count:
    #                 bi_pos_count[(review[idx_word], review[idx_word+1])] = bi_pos_count[(review[idx_word], review[idx_word+1])] + 1
    #             else:
    #                 bi_pos_count[(review[idx_word], review[idx_word+1])] = 1


    #         else:
    #             # unigram
    #             if (review[idx_word] in uni_neg_count):
    #                 uni_neg_count[review[idx_word]] += 1
    #             else:
    #                 uni_neg_count[review[idx_word]] = 1

                
    #             # bigram
    #             if (review[idx_word], review[idx_word+1]) in bi_neg_count:
    #                 bi_neg_count[(review[idx_word], review[idx_word+1])] = bi_neg_count[(review[idx_word], review[idx_word+1])] + 1
    #             else:
    #                 bi_neg_count[(review[idx_word], review[idx_word+1])] = 1
      
        
    #     # unigram last word in review
    #     if (train_labels[idx_label] == 1):
    #         if (review[-1] in uni_pos_count):
    #             uni_pos_count[review[-1]] += 1
    #         else:
    #             uni_pos_count[review[-1]] = 1

    #     else:
    #         if (review[-1] in uni_neg_count):
    #             uni_neg_count[review[-1]] += 1
    #         else:
    #             uni_neg_count[review[-1]] = 1

    #     idx_label += 1


    neg_prior = 1 - pos_prior
   
    uni_pos_num_words_total = sum(uni_pos_count.values())
    uni_neg_num_words_total = sum(uni_neg_count.values())
    bi_pos_num_words_total = sum(bi_pos_count.values())
    bi_neg_num_words_total = sum(bi_neg_count.values())
    
    print(uni_pos_num_words_total)
    print(uni_neg_num_words_total)
    print(bi_pos_num_words_total)
    print(bi_neg_num_words_total)
    
    ### Development Phase ### 
    yhats = []
    uni_pos_posterior = 0
    uni_neg_posterior = 0
    bi_pos_posterior = 0
    bi_neg_posterior = 0

    len_uni_pos = len(uni_pos_count)
    len_uni_neg = len(uni_neg_count)
    len_bi_pos = len(bi_pos_count)
    len_bi_neg = len(bi_neg_count)


    pos_mix_model = 0
    neg_mix_model = 0
    

    for doc in tqdm(dev_set,disable=silently):
        # unigram
        for word in doc:
            if word in uni_pos_count:
                uni_pos_posterior = uni_pos_posterior + math.log((uni_pos_count[word] + unigram_laplace) / (uni_pos_num_words_total + unigram_laplace * (len_uni_pos + 1)))
            else:
                uni_pos_posterior = uni_pos_posterior + math.log(unigram_laplace / (uni_pos_num_words_total + unigram_laplace * (len_uni_pos + 1)))


            if word in uni_neg_count:
                uni_neg_posterior = uni_neg_posterior + math.log((uni_neg_count[word] + unigram_laplace) / (uni_neg_num_words_total + (unigram_laplace * (len_uni_neg + 1))))
            else:
                uni_neg_posterior = uni_neg_posterior + math.log(unigram_laplace / (uni_neg_num_words_total + (unigram_laplace * (len_uni_neg + 1))))
    


        for idx_word in range(len(doc)-1):
            # bigram posteriors
            if (doc[idx_word], doc[idx_word+1]) in bi_pos_count:
                bi_pos_posterior += math.log((bi_pos_count[(doc[idx_word], doc[idx_word+1])] + bigram_laplace) / (bi_pos_num_words_total + bigram_laplace * (len_bi_pos + 1)))
            else:
                bi_pos_posterior += math.log(bigram_laplace / (bi_pos_num_words_total + bigram_laplace * (len_bi_pos + 1)))


            if (doc[idx_word], doc[idx_word+1]) in bi_neg_count:
                bi_neg_posterior += math.log((bi_neg_count[(doc[idx_word], doc[idx_word+1])] + bigram_laplace) / (bi_neg_num_words_total + (bigram_laplace * (len_bi_neg + 1))))
            else:
                bi_neg_posterior += math.log(bigram_laplace / (bi_neg_num_words_total + (bigram_laplace * (len_bi_neg + 1))))



        uni_pos_posterior = uni_pos_posterior + math.log(pos_prior)
        uni_neg_posterior = uni_neg_posterior + math.log(neg_prior)
        bi_pos_posterior = bi_pos_posterior + math.log(pos_prior)       
        bi_neg_posterior = bi_neg_posterior + math.log(neg_prior)       

        pos_mix_model = (1.0 - bigram_lambda) * uni_pos_posterior + bigram_lambda * bi_pos_posterior
        neg_mix_model = (1.0 - bigram_lambda) * uni_neg_posterior + bigram_lambda * bi_neg_posterior


        if (pos_mix_model > neg_mix_model):
            yhats.append(1)

        else:
            yhats.append(0)

        uni_pos_posterior = 0
        uni_neg_posterior = 0
        bi_pos_posterior = 0
        bi_neg_posterior = 0
        pos_mix_model = 0
        neg_mix_model = 0





    # for doc in tqdm(dev_set,disable=silently):

    #     for idx_word in range(len(doc)-1):
    #         # unigram posteriors
    #         if (doc[idx_word] in uni_pos_count):
    #             uni_pos_posterior = uni_pos_posterior +  math.log((uni_pos_count[doc[idx_word]] + unigram_laplace) / (uni_pos_num_words_total + unigram_laplace * (len_uni_pos+1)))
    #         else:
    #             uni_pos_posterior = uni_pos_posterior +  math.log(unigram_laplace / uni_pos_num_words_total + unigram_laplace * (len_uni_pos+1))

    #         if (doc[idx_word] in uni_neg_count):
    #             uni_neg_posterior = uni_neg_posterior +  math.log((uni_neg_count[doc[idx_word]] + unigram_laplace) / (uni_neg_num_words_total + unigram_laplace * (len_uni_neg+1)))
    #         else:
    #             uni_neg_posterior = uni_neg_posterior +  math.log(unigram_laplace / uni_neg_num_words_total + unigram_laplace * (len_uni_neg+1))


    #         # bigram posteriors
    #         if (doc[idx_word], doc[idx_word+1]) in bi_pos_count:
    #             bi_pos_posterior += math.log((bi_pos_count[(doc[idx_word], doc[idx_word+1])] + bigram_laplace) / (bi_pos_num_words_total + bigram_laplace * (len_bi_pos + 1)))
    #         else:
    #             bi_pos_posterior += math.log(bigram_laplace / (bi_pos_num_words_total + bigram_laplace * (len_bi_pos + 1)))


    #         if (doc[idx_word], doc[idx_word+1]) in bi_neg_count:
    #             bi_neg_posterior += math.log((bi_neg_count[(doc[idx_word], doc[idx_word+1])] + bigram_laplace) / (bi_neg_num_words_total + (bigram_laplace * (len_bi_neg + 1))))
    #         else:
    #             bi_neg_posterior += math.log(bigram_laplace / (bi_neg_num_words_total + (bigram_laplace * (len_bi_neg + 1))))


    #     # unigram posterior of the last word in doc
    #     if doc[-1] in uni_pos_count:
    #         uni_pos_posterior = uni_pos_posterior + math.log((uni_pos_count[doc[-1]] + unigram_laplace) / (uni_pos_num_words_total + unigram_laplace * (len_uni_pos+1)))
    #     else:
    #         uni_pos_posterior = uni_pos_posterior +  math.log(unigram_laplace / uni_pos_num_words_total + unigram_laplace * (len_uni_pos+1))


    #     if doc[-1] in uni_neg_count:
    #         uni_neg_posterior = uni_neg_posterior +  math.log((uni_neg_count[doc[-1]] + unigram_laplace) / (uni_neg_num_words_total + unigram_laplace * (len_uni_neg+1)))
    #     else:
    #         uni_neg_posterior = uni_neg_posterior +  math.log(unigram_laplace / uni_neg_num_words_total + unigram_laplace * (len_uni_neg+1))


    #     uni_pos_posterior += math.log(pos_prior)
    #     uni_neg_posterior += math.log(neg_prior)
    #     bi_pos_posterior += math.log(pos_prior)       
    #     bi_neg_posterior += math.log(neg_prior)       

    #     pos_mix_model = (1.0 - bigram_lambda) * uni_pos_posterior + bigram_lambda * bi_pos_posterior
    #     neg_mix_model = (1.0 - bigram_lambda) * uni_neg_posterior + bigram_lambda * bi_neg_posterior


    #     if (pos_mix_model > neg_mix_model):
    #         yhats.append(1)

    #     else:
    #         yhats.append(0)

    #     uni_pos_posterior = 0
    #     uni_neg_posterior = 0
    #     bi_pos_posterior = 0
    #     bi_neg_posterior = 0
    #     pos_mix_model = 0
    #     neg_mix_model = 0


    # Keep this in the provided template
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    # yhats = []
    # for doc in tqdm(dev_set,disable=silently):
    #     yhats.append(-1)
    return yhats

