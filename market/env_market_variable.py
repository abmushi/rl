"""

"""
from rl_glue import BaseEnvironment
import numpy as np
import sys
import constant as constant

class VariableMarketEnvironment(BaseEnvironment):

    def __init__(self):
        """Declare environment variables."""
        self._x = None
        self._y = None

        # place to buy item
        self._source = None

        # place to sell item
        self._market = None

        self.last_state = None
        self.map = None

        self.count = 0
        print("Variable Market Environment")

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        """
        self._x = constant._x
        self._y = constant._y

        self._source = [np.random.randint(self._x),np.random.randint(self._y)]
        self._market = [np.random.randint(self._x),np.random.randint(self._y)]

        self.last_state = [np.random.randint(self._x),np.random.randint(self._y),0]
        self.map = np.zeros((self._x,self._y))

        self.map[self._source[0],self._source[1]] = constant.TYPE_SOURCE
        self.map[self._market[0],self._market[1]] = constant.TYPE_MARKET

        self.count = 0


        # a = [[0,0,0,1,0,0,0,1,1,0],[0,1,0,0,0,2,0,0,0,0],[0,1,1,1,1,1,1,0,1,10],[0,0,1,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,1,1,1],[0,1,1,0,1,1,0,0,0,0],[0,0,1,0,0,0,0,0,1,1]]


        # for i in range(self._x):
        #     for j in range(self._y):
        #         if a[i][j] == 1:
        #             self.map[i,j] = constant.TYPE_WALL
        #         elif a[i][j] == 2:
        #             self.map[i,j] = constant.TYPE_CLIFF
        #         elif a[i][j] == 10:
        #             self.map[i,j] = constant.TYPE_GOAL

    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        """
        self.last_state = [np.random.randint(self._x),np.random.randint(self._y),0]

        # initialize source and market positions occasionally
        if self.count % 5000 == 0:
            self._source = [np.random.randint(self._x),np.random.randint(self._y)]
            self._market = [np.random.randint(self._x),np.random.randint(self._y)]
            self.map = np.zeros((self._x,self._y))

            self.map[self._source[0],self._source[1]] = constant.TYPE_SOURCE
            self.map[self._market[0],self._market[1]] = constant.TYPE_MARKET

        return self.last_state

    def env_step(self, action):
        """
        Arguments: action - integer[up:0, right:1, down:2, left:3]
        Returns: reward - float, state - numpy array - terminal - boolean
        """
        next_state = list(self.last_state)

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
            self.last_state[2] = 0
            next_state = self.last_state
            return -1, next_state, False

        # check type of map chip
        next_state_type = self.map[next_state[0],next_state[1]]
        self.count += 1

        # reach source
        if next_state_type == constant.TYPE_SOURCE:
            if self.last_state[2] < constant._max_num_of_items-1:
                next_state[2] += 1
                self.last_state = next_state
                return 0, next_state, False
            else:
                return 0, next_state, False

        # reach market
        if next_state_type == constant.TYPE_MARKET:
            if self.last_state[2] > 0:
                reward = next_state[2]
                next_state[2] = 0
                self.last_state = next_state
                return reward, next_state, True
            else:
                return 0, next_state, False

        # wall
        if next_state_type == constant.TYPE_WALL:
            return 0, self.last_state, False

        # cliff
        if next_state_type == constant.TYPE_CLIFF:
            return -100, self.last_state, True

        self.last_state = next_state

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
