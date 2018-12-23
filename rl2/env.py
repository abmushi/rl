"""

"""
from rl_glue import BaseEnvironment
import numpy as np
import sys
import constant as constant

class GridEnvironment(BaseEnvironment):

    def __init__(self):
        """Declare environment variables."""
        self._x = None
        self._y = None
        self._goal = None
        self.s = None
        self.map = None
        

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        """
        self._x = constant._x
        self._y = constant._y
        
        self._goal = [3,5]
        self.s = [0,0,0]
        self.map = np.zeros((self._x,self._y))
        self.map[self._goal[0],self._goal[1]] = constant.TYPE_GOAL

    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        """
        self.s = [np.random.randint(self._x),np.random.randint(self._y),0]
        return self.s

    def env_step(self, action):
        """
        Arguments: action - integer[up:0, right:1, down:2, left:3]
        Returns: reward - float, state - numpy array - terminal - boolean
        """

        next_state = list(self.s)

        if action == 0:
            next_state[1] -= 1
        elif action == 1:
            next_state[0] += 1
        elif action == 2:
            next_state[1] += 1
        elif action == 3:
            next_state[0] -= 1
        else:
            sys.exit("error: invalid action: ",action)

        # off-grid action
        if next_state[0] < 0 or next_state[0] >= self._x or next_state[1] < 0 or next_state[1] >= self._y:
            return -1, self.s, True

        next_state_type = self.map[next_state[0],next_state[1]]

        # reach goal
        if next_state_type == constant.TYPE_GOAL:
            return 1, self.s, True

        if next_state_type == constant.TYPE_WALL:
            return -1, self.s, False

        if next_state_type == constant.TYPE_CLIFF:
            return -100, self.s, True

        self.s = next_state
        # else
        return 0, next_state, False

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        """
        if in_message == 'GOAL':
            return self._goal
        elif in_message == 'MAP':
            return self.map
        else:
            return sys.exit("error: invalid message: ",in_message)
