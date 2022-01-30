# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()

        
        # This gave 40 me on gradescope: w/ 0.7827 and 0.7570
        self.model = nn.Sequential(
                        nn.Conv2d(3, 20, 5),    #
                        nn.ReLU(),                  # activation function

                        # nn.Conv2d(16,(3,3),activation = "relu" , input_shape = (180,180,3)),
                        nn.MaxPool2d(2, 2),
                    
                        # nn.Linear(28, 14),    
                        # nn.BatchNorm2d(20),         #
                        nn.Conv2d(20, 20, 10),      #
                        nn.Flatten(1),
                        nn.ReLU(),                  # activation function
                        # nn.Tanh(),
                        # nn.Dropout(p = 0.2),
                        nn.Dropout(p = 0.1),
                        nn.Linear(500, 50),
                        # nn.Tanh(),
                        nn.ReLU(),
                        # nn.Dropout(p = 0.1),
                        nn.Linear(50, 25), # 25
                        # nn.Tanh(),
                        nn.ReLU(),
                        nn.Linear(25, out_size)
                    )

        
        '''
        Try improving this for leaderboard(?)
        self.model = nn.Sequential(
                        nn.Conv2d(3, 20, 5),    #
                        nn.ReLU(),                  # activation function

                        # nn.Conv2d(16,(3,3),activation = "relu" , input_shape = (180,180,3)),
                        nn.MaxPool2d(2, 2),

                        nn.Conv2d(20, 30, 5),
                        nn.ReLU(),
                   
                        # nn.Linear(28, 14),    
                        # nn.BatchNorm2d(20),         #
                        nn.Conv2d(30, 36, 5),      #
                        
                        # nn.Flatten(1),
                        nn.ReLU(),                  # activation function
                        nn.MaxPool2d(2, 2),
                        # nn.Tanh(),
                        # nn.Dropout(p = 0.2),
                        nn.Dropout(p = 0.1),
                        nn.Linear(500, 50),
                        # nn.Tanh(),
                        nn.ReLU(),
                        # nn.Dropout(p = 0.1),
                        nn.Linear(50, 25), # 25
                        # nn.Tanh(),
                        nn.ReLU(),
                        nn.Linear(25, out_size)


                        # three conv and one or two linear
                        # don't do dropout
                        # make sure you're Reluing after every layer
                        # MaxPool2d after ever conv
                    )
        '''

        self.loss_fn = loss_fn
        self.lrate = lrate

        # optimization
        self.op1 = optim.SGD(self.model.parameters(),lr=lrate, momentum=0.4, weight_decay = 1e-5)  

        # raise NotImplementedError("You need to write this part!")
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

        newShape = int(np.sqrt(x.size(1)/3))
        new_x = torch.reshape(x,(int(x.size(0)),3,newShape,newShape))
        # newx = x.view([-1, 3, 32, 32])
        

        return self.model(new_x)

        # raise NotImplementedError("You need to write this part!")
        # return torch.ones(x.shape[0], 1)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """

        self.op1.zero_grad()
        yhat = self.forward(x)
        loss_value = self.loss_fn(yhat, y)

        loss_value.backward()
        self.op1.step()

        return loss_value.detach().cpu().numpy()


        # raise NotImplementedError("You need to write this part!")
        # return 0.0

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    # print(train_set.size())
    # img_size = train_set.size(1)
    # print(img_size)

    # N1 = NeuralNet(lrate = 0.1, loss_fn = nn.CrossEntropyLoss(), in_size = 3072, out_size = 4)
    N1 = NeuralNet(lrate = 0.1, loss_fn = nn.CrossEntropyLoss(), in_size = train_set.size(1), out_size = 4) #, in_size = img_size, out_size = 4)
    arr1 = get_dataset_from_arrays(train_set,train_labels)

    train_dataloader = DataLoader(arr1, batch_size, shuffle = False, num_workers = 1) # num_workers : num of CPU

    loss_list = []

    # normalizing trian_set
    for i in range(len(train_set)):
        curr_train_data = train_set[i] 
        mean = curr_train_data.mean()
        std = curr_train_data.std()

        train_set[i] = (curr_train_data - mean) / std


    # normalizing dev_set
    for i in range(len(dev_set)):
        curr_dev_data = dev_set[i] 
        mean = curr_dev_data.mean()
        std = curr_dev_data.std()

        dev_set[i] = (curr_dev_data - mean) / std
   
        
    # Training
    for i in range(epochs):
        for batch_map in train_dataloader:
            data_batch = batch_map['features'] 
            label_batch = batch_map['labels'] 

            loss_list.append(N1.step(data_batch, label_batch))
            pass; 
        pass
    

    arr2 = N1(dev_set)
    argmax_arr = np.ndarray(shape = (len(arr2)))


    for i in range(len(arr2)):
        argmax_arr[i] = (np.argmax(arr2[i].detach().cpu().numpy()))


    return loss_list, argmax_arr.astype(int), N1



    # raise NotImplementedError("You need to write this part!")
    # return [],[],None
