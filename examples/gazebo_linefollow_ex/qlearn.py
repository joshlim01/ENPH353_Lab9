import random
import pickle
import numpy as np
import math
from re import A


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

        states = ['1000000000', 
                '0100000000', 
                '0010000000', 
                '0001000000', 
                '0000100000', 
                '0000010000', 
                '0000001000', 
                '0000000100', 
                '0000000010', 
                '0000000001']

        def state_to_string(state, action):
            tuple = (state, action)
            return tuple

        for i in range(10):
            for j in range(3):
                self.q[state_to_string(states[i], j)] = 1


    def loadQ(self, filename):
        '''
        Load the Q state-action values from a pickle file.
        '''
        # TODO: Implement loading Q values from pickle file.
        # source: https://www.geeksforgeeks.org/understanding-python-pickling-example/
        dbfile = open(filename, 'rb')     
        db = pickle.load(dbfile)
        for keys in db:
            self.q[keys] = db[keys]
        dbfile.close()

        print("Loaded file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        '''
        Save the Q state-action values in a pickle file.
        '''
        # TODO: Implement saving Q values to pickle and CSV files.
        # source: https://www.geeksforgeeks.org/understanding-python-pickling-example/
        db = {}
        for keys in self.q:
            db[keys] = self.q[keys]
        
        # Its important to use binary mode
        dbfile = open(filename, 'wb')
        
        # source, destination
        pickle.dump(db, dbfile)                     
        dbfile.close()

        print("Wrote to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action
        
        if random.randrange(100) > self.epsilon * 100:
            ret = 0
            maxreward = -4758132194

            for keys in self.q:
                if keys[0] == state:
                    if self.q[keys] > maxreward:
                        maxreward = self.q[keys]
                        ret = keys[1]
        else:
            ret = random.randrange(3)

        if return_q:
            return (ret, maxreward)
        else:
            return ret

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)
        oldvalue = self.q[(state1,action1)]
        maxterm = -4758132194

        for keys in self.q:
            if keys[0] == state2:
                if self.q[keys] > maxterm:
                    maxterm = self.q[keys]

        addition = self.alpha * (reward + self.gamma * maxterm - self.q[(state1, action1)])

        # THE NEXT LINE NEEDS TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE
        self.q[(state1,action1)] = oldvalue + addition
