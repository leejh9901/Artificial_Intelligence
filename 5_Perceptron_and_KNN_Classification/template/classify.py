# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters

    # single class perceptron
    # if y_cal is y_correct             y_cal, y_corerct -> sign ? 0/1 ? or f_values ?  ** The only possible values for y_cal and y_correct are 1 and -1.
    #   Do Nothing
    # if y_cal is not y_correct
    #       w_i = w_i + learning_rate * y_correct * x_i     where x_i is the input, w_i is the weight
    #   OR  w_i = w_i + learning_rate * (y_correct - y_cal) * x_i



    W = np.zeros(len(train_set[0]))      # array of weights
    b = 0      # bias parameters

    currSum = 0
    # idx_pic = 0
    isAnimal = False    # 1 if there's an animal; 0 if there's no animal

# Fill in W    
    for i in range(max_iter):
        for j in range(len(train_set)):
            image = train_set[j]
            label = train_labels[j]
        # for image in train_set:
        #     label  = train_labels[idx_pic]

            currSum = np.dot(W, image) + b
            isAnimal = currSum > 0
            # if np.sign(currSum) > 0 :
            #     isAnimal = True
            #     # pass
            # else:
            #     isAnimal = False
            #     pass
            
            #
            if isAnimal != label:
                if label:
                    new_learning_rate = learning_rate
                else:
                    new_learning_rate = -learning_rate
                
                
                add_term = np.multiply(new_learning_rate, image)
                W = np.add(W, add_term)
                b += new_learning_rate

                pass
            

            # idx_pic += 1


            pass


    # return
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set

    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)

    P = []  # predicted label

    for image in dev_set:
        currSum = np.dot(W, image) + b
        isAnimal = currSum > 0

        P.append(isAnimal)
        pass
    
    return P
    # return []

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here

    # final_label = np.array([])    # calculated labels for dev_set
    # dist = np.array([])           # array of tuples (dist, train_labels)          # np.zeros(len(train_set[0]))
    # temp_dist = 0.0
    P = []      # predicted labels

    for image in dev_set:
        dists = list() 
        for i in range(len(train_set)):

            temp_dist = np.linalg.norm(train_set[i] - image)
            dists.append( (temp_dist, train_labels[i]) ) 

            pass; 

        list.sort(dists)

        dists = dists[:k]


        count = dict()  # key: lable (True/False), value: # of labels

        max_item = (0, None)    # (num of occurances, label)
        isTied = False
        for distance, label in dists:
            count[label] = count.get(label, 0) + 1

            if count[label] > max_item[0]:
                max_item = (count[label], label)
                isTied = False
            elif count[label] == max_item[0]:
                isTied = True
            pass


        if isTied:
            P.append(False)
        else:
            P.append(max_item[1])


    return P
