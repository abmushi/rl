from rl_glue import BaseAgent
import numpy as np
import sys
import constant as constant

class DynaQPlusAgent(BaseAgent):
    """
    Dyna-Q+ Agent
    """

    def __init__(self):
        """Declare agent variables."""
        self._x = None
        self._y = None
        self._max_num_of_items = None

        self.gamma = None
        self.episilon = None
        self.alpha = None

        self.Q = None
        self.plan = None
        self.n = None
        self.tau = 0
        
        self.last_action = None
        self.last_state = None

        self.count = 0

        print('Dyna-Q+ Agent')

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        """
        self._x = constant._x
        self._y = constant._y
        self._max_num_of_items = constant._max_num_of_items


        self.gamma = 0.9        #   discount rate
        self.episilon = 0.05     #   epsilon-greedy
        self.alpha = 0.5

        self.Q = np.zeros((self._x,self._y,self._max_num_of_items,4))
        self.plan = {}
        self.n = 10
        self.tau = 0
        self.kappa = 0.001

        self.last_action = 0
        self.last_state = [0,0,0]

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

        self.tau += 1

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

        last_Q = self.Q[self.last_state[0],self.last_state[1],self.last_state[2],self.last_action]

        # optimistic extimate on next state value
        target = reward + self.gamma*np.max(self.Q[state[0],state[1],state[2]])

        self.Q[self.last_state[0],self.last_state[1],self.last_state[2],self.last_action] += self.alpha*(target - last_Q)

        # add to model plan
        self.plan[self.last_state[0],self.last_state[1],self.last_state[2],self.last_action] = [reward,state,self.tau]

        # planning
        plan_keys = list(self.plan)
        for i in range(min(self.n,len(plan_keys))):
            key_index = np.random.randint(len(plan_keys))
            key = plan_keys[key_index]
            s0 = [key[0],key[1],key[2]]
            a0 = key[3]
            r = self.plan[key][0]
            s1 = self.plan[key][1]
            t = self.tau - self.plan[key][2]
            # G = r + self.gamma*np.max(self.Q[s1[0],s1[1],s1[2]])
            G = r + self.kappa*np.sqrt(t) + self.gamma*np.max(self.Q[s1[0],s1[1],s1[2]])
            self.Q[s0[0],s0[1],s0[2],a0] += self.alpha*(G - self.Q[s0[0],s0[1],s0[2],a0])

        # choose action based on current policy
        rnd = np.random.random()
        if rnd < self.episilon:
            #   epsilon-case: random
            next_action = np.random.randint(0,4)
        else:
            #   greedy-case: optimal policy
            next_action = np.random.choice(np.flatnonzero(self.Q[state[0],state[1],state[2]] == self.Q[state[0],state[1],state[2]].max()))

        self.last_action = next_action
        self.last_state = state

        self.tau += 1

        return next_action


    def agent_end(self, reward):
        """
        Arguments:
            reward - floating point
        Returns:
            Nothing
        """
        self.Q[self.last_state[0],self.last_state[1],self.last_state[2],self.last_action] += self.alpha*(reward - self.Q[self.last_state[0],self.last_state[1],self.last_state[2],self.last_action])
        if reward > 0:
            self.count += reward

    def agent_message(self, in_message):
        """
        Arguments:
            in_message - string
        Returns:
            The value function as a list.
        """
        if in_message == 'V':
            return np.max(self.Q, axis=3)
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