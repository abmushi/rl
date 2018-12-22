"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
import math
import constant as C

class WindyAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.N = None
        self.gamma = 1.0
        self.epsilon = 0.1
        self.alpha = 0.5
        self.Q = None
        self.state = None
        self.last_action = None

        self.count = 0

    def _take_action(self,state):
        if C.ACTION_TYPE == 1:
            action_set = [0,1,2,3,4,5,6,7]
            return action_set
        elif C.ACTION_TYPE == 2:
            action_set = [0,1,2,3,4,5,6,7,8]
            return action_set
        else:
            action_set = [0,1,2,3]
            return action_set
        

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.Q = np.zeros((C.height,C.width,C.ACTION_SIZE))    
        
        self.state = [0,0]
        self.last_action = 0

        self.count = 0

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """

        #   initialize state
        self.state = state

        #   get valid action set.
        action_set = self._take_action(state)

        rnd = np.random.random()

        #   epsilon case.
        #   take random action.
        if rnd < self.epsilon:
            return action_set[np.random.randint(len(action_set))]

        #   if not, greedy policy
        next_action = 0
        max_q = -math.inf
        for action in action_set:
            # print(state)
            # print(action)
            if self.Q[(state[0],state[1],action)] > max_q:
                next_action = action
                max_q = self.Q[(state[0],state[1],action)]

        self.last_action = next_action
        return next_action

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        """

        #   set current state
        last_state = self.state
        self.state = state

        state = [int(state[0]),int(state[1])]
        last_state = [int(last_state[0]),int(last_state[1])]

        #   get valid action set.
        action_set = self._take_action(state)

        #   Q(S,A)
        q1 = self.Q[(last_state[0],last_state[1],self.last_action)]

        rnd = np.random.random()

        #   epsilon case.
        #   take random action.
        if rnd < self.epsilon:
            next_action = action_set[np.random.randint(len(action_set))]

            #   Q(S',A')
            q2 = self.Q[(state[0],state[1],next_action)]
            self.Q[(last_state[0],last_state[1],self.last_action)] += self.alpha*(reward + self.gamma*q2 - q1)
            self.last_action = next_action
            return next_action

        #   if not, greedy policy
        next_action = 0
        max_q = -math.inf
        for action in action_set:
            if self.Q[(state[0],state[1],action)] > max_q:
                next_action = action
                max_q = self.Q[(state[0],state[1],action)]

        #   Q(S',A')
        q2 = self.Q[(state[0],state[1],next_action)]
        self.Q[(last_state[0],last_state[1],self.last_action)] += self.alpha*(reward + self.gamma*q2 - q1)

        self.last_action = next_action

        return next_action


    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        """
        self.count += 1


    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'COUNT':
            return self.count
        elif in_message == 'STATE':
            return self.state
        else:
            return "I dont know how to respond to this message!!"
