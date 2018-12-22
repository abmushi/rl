from rl_glue import BaseAgent
import numpy as np
import sys
import constant as constant

class TDAgent(BaseAgent):
    """
    TD Agent
    """

    def __init__(self):
        """Declare agent variables."""
        self._x = None
        self._y = None
        self.gamma = None
        self.episilon = None
        self.alpha = None
        self.V = None
        
        self.last_action = None
        self.state = None

        self.count = 0

        print('TD Agent - random policy')

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        """
        self._x = constant._x
        self._y = constant._y
        self.gamma = 0.9        #   discount rate
        self.episilon = 0.1     #   epsilon-greedy
        self.alpha = 0.9

        self.V = np.zeros((self._x,self._y))

        self.last_action = 0
        self.last_state = [0,0]

    def agent_start(self, state):
        """
        Arguments:
            state - numpy array
        Returns:
            action - integer
        """
        self.last_state = state

        next_action = np.random.randint(0,4)
        self.last_action = next_action

        #   first action
        return next_action

    def agent_step(self, reward, state):
        """
        Arguments:
            reward: R       ---- floting point
                            ---- reward
            state: S'       ---- [integer,integer]
                            ---- next state
        Returns: action - integer
            action: A'      ---- integer
                            ---- next action
        """
        
        v = self.V[self.last_state[0],self.last_state[1]]

        target = reward + self.gamma*self.V[state[0],state[1]]

        self.V[self.last_state[0],self.last_state[1]] += self.alpha*(target - v)

        # choose action based on current policy     
        next_action = np.random.randint(0,4)

        self.last_action = next_action
        self.last_state = state

        return next_action


    def agent_end(self, reward):
        """
        Arguments:
            reward - floating point
        Returns:
            Nothing
        """
        self.V[self.last_state[0],self.last_state[1]] += self.alpha*(reward - self.V[self.last_state[0],self.last_state[1]])
        if reward == 1:
            self.count += 1

    def agent_message(self, in_message):
        """
        Arguments:
            in_message - string
        Returns:
            The value function as a list.
        """
        if in_message == 'V':
            return self.V
        elif in_message == 'STATE':
            return self.last_state
        elif in_message == 'COUNT':
            count = self.count
            self.count = 0
            return count
        else:
            return sys.exit("error: invalid message: ",in_message)
