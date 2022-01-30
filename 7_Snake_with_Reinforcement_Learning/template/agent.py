import numpy as np
import utils
from math import inf 

# GROUND_SIZE = (utils.DISPLAY_SIZE - 2 * utils.WALL_SIZE)/ utils.GRID_SIZE # the number of grid on the board

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0 # previous point
        self.s = None   # previous state
        self.a = None   # previous action
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO: write your function here

        '''
        self.Q[food_dir_x][food_dir_y]...
        self.Q[(food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, 
                    adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)]
        state = generate_state(self, environment)
        then self.Q[state]
            Like this...
        
        state = self.generate_state(environment)
        self.Q[state]
        self.N[state][action]
        '''
        
        # Testing (Not training)
        if self.train is False:
            return np.argmax(self.Q[s_prime])
            
        
        if (self.s == None) and (self.a == None) and (points == 0):
            argmax_a = self.findOptimalAction(s_prime)
            self.s = s_prime
            self.a = argmax_a
            return argmax_a
        

        ### Reward ###
        reward = -1 if dead else(1 if self.points < points else -0.1)
        
        self.points = points
        

        ### Update N ###
        self.N[self.s][self.a] += 1


        old_Q = self.Q[self.s][self.a]
        old_N = self.N[self.s][self.a]
       

        ### Update Q ###
        learning_rate = self.C / (self.C + old_N)
        self.Q[self.s][self.a] = old_Q + learning_rate * (reward + self.gamma * np.max(self.Q[s_prime]) - old_Q)
        
        if dead:
            self.reset()
            return utils.RIGHT
        

        ### Update the new state and action ###
        argmax_a = self.findOptimalAction(s_prime)
        self.s = s_prime
        self.a = argmax_a
        
       
        return argmax_a
    

    def findOptimalAction(self, state):
        # tie: priority order RIGHT (3) > LEFT (2) > DOWN (1) > UP (0)
        max_Q = -inf
        argmax_a = -1
      
        for action in [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]:
            if self.train and (self.N[state][action] < self.Ne):
                curr_Q = 1
            else:
                curr_Q = self.Q[state][action]

            if curr_Q > max_Q:
                max_Q = curr_Q
                argmax_a = action


        return argmax_a


    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        '''
            State: 
                (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, 
                    adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right).

            environment:
                a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
                All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        snake_head_x, snake_head_y, snake_body, food_x, food_y = environment
      
        # [food_dir_x, food_dir_y]
        food_dir_x = 0 if snake_head_x == food_x else (1 if snake_head_x > food_x else 2)
        food_dir_y = 0 if snake_head_y == food_y else (1 if snake_head_y > food_y else 2)

        
        # [adjoining_wall_x, adjoining_wall_y]
        # [adjoining_wall_x, adjoining_wall_y] = [0, 0] can also occur when the snake runs out of the board boundaries.
        if snake_head_x == 0 or snake_head_x == utils.DISPLAY_SIZE - utils.WALL_SIZE:
            adjoining_wall_x = 0
        elif (snake_head_x == utils.WALL_SIZE):
            adjoining_wall_x = 1
        elif (snake_head_x == utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE):
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        if snake_head_y == 0 or snake_head_y == utils.DISPLAY_SIZE - utils.WALL_SIZE:
            adjoining_wall_y = 0
        elif (snake_head_y == utils.WALL_SIZE):
            adjoining_wall_y = 1
        elif (snake_head_y == utils.DISPLAY_SIZE - utils.WALL_SIZE - utils.GRID_SIZE):
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0


        # [adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]
        adjoining_body_top = 1 if ((snake_head_x, snake_head_y - utils.GRID_SIZE) in snake_body) else 0
        adjoining_body_bottom = 1 if ((snake_head_x, snake_head_y + utils.GRID_SIZE) in snake_body) else 0
        adjoining_body_left = 1 if ((snake_head_x - utils.GRID_SIZE, snake_head_y) in snake_body) else 0
        adjoining_body_right = 1 if ((snake_head_x + utils.GRID_SIZE, snake_head_y) in snake_body) else 0


        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, 
                    adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)


