from rl_glue import BaseAgent
import numpy as np
import sys
import constant as constant

class ExpectedSARSAgent(BaseAgent):
    """
    Expected SARSA Agent
    """

    def __init__(self):
        """Declare agent variables."""
        self._x = None
        self._y = None
        self.gamma = None
        self.episilon = None
        self.alpha = None
        self.R = None
        self.Q = None
        
        self.last_action = None
        self.last_state = None

        self.count = 0

        print('Expected SARSA Agent')

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

        self.Q = np.zeros((self._x,self._y,4))

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

        # choose action based on current policy
        rnd = np.random.random()
        if rnd < self.episilon:
            #   epsilon-case: random
            next_action = np.random.randint(0,4)
        else:
            #   greedy-case: optimal policy
            next_action = np.random.choice(np.flatnonzero(self.Q[state[0],state[1]] == self.Q[state[0],state[1]].max()))

        q = self.Q[self.last_state[0],self.last_state[1],self.last_action]

        # expected Q value
        target = reward + self.gamma*np.mean(self.Q[state[0],state[1]])

        self.Q[self.last_state[0],self.last_state[1],self.last_action] += self.alpha*(target - q)

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
        self.Q[self.last_state[0],self.last_state[1],self.last_action] += self.alpha*(reward - self.Q[self.last_state[0],self.last_state[1],self.last_action])
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
            return np.max(self.Q, axis=2)
        elif in_message == 'Q':
            return self.Q
        elif in_message == 'STATE':
            return self.last_state
        elif in_message == 'COUNT':
            count = self.count
            self.count = 0
            return count
        else:
            return sys.exit("error: invalid message: ",in_message)
