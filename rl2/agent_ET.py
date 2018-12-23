from rl_glue import BaseAgent
import numpy as np
import sys
import constant as constant
import tiles3 as tiles3

class ETAgent(BaseAgent):
    """
        Eligibility Trace
        Sarsa(lambda) Agent with tile coding
    """

    def __init__(self):
        """Declare agent variables."""
        self._x = None
        self._y = None
        self._z = None

        self.num_of_tilings = 8
        self.dimension = 1024
        self.iht = tiles3.IHT(self.dimension)

        self.gamma = 0.9        #   discount rate
        self.epsilon = 0.05    #   epsilon-greedy
        self.alpha = 0.5 / self.num_of_tilings

        #   weight vector
        self.w = None

        self.z = None           #   z for eligibility trace
        self.lmbda = 0.9         #   lambda for eligibility trace
        
        self.last_action = None
        self.last_state = None

        self.Q = None

        self.count = 0

        print('Sarsa(lambda) Agent with tile coding')

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        """

        self._x = constant._x
        self._y = constant._y
        self._z = constant._z

        self.w = np.zeros(self.dimension)
        self.z = np.zeros(self.dimension)

        self.Q = np.zeros((self._x,self._y,self._z,4))

        self.last_action = 0
        self.last_state = [0,0,0]

    def agent_start(self, state):
        """
        Arguments:
            state - numpy array
        Returns:
            action - integer
        """
        self.z = np.zeros(self.dimension)

        #   initialize state
        self.last_state = state

        #   greedy start
        rnd = np.random.rand()
        next_action = 0
        if rnd < self.epsilon:
            next_action = np.random.randint(4)
        else:
            next_action, _ = self.greedy_action(state)
        
        self.last_action = next_action

        #   first action
        return next_action

    def feature(self,state,action):
        s1 = (state[0])/constant._x * 10.0
        s2 = (state[1])/constant._y * 10.0
        s3 = state[2]*10
        return tiles3.tiles(self.iht, self.num_of_tilings, [s1,s2,s3],[action])

    def greedy_action(self,state):

        # for each action 0,1,2,3 get feature
        f0 = self.feature(state, 0)
        f1 = self.feature(state, 1)
        f2 = self.feature(state, 2)
        f3 = self.feature(state, 3)

        # convert feature to feacture vector
        x0 = np.zeros(self.dimension)
        x1 = np.zeros(self.dimension)
        x2 = np.zeros(self.dimension)
        x3 = np.zeros(self.dimension)

        for i in range(len(f0)):
            x0[f0[i]] = 1
            x1[f1[i]] = 1
            x2[f2[i]] = 1
            x3[f3[i]] = 1

        # get q value
        q = np.zeros(4)
        q[0] = np.dot(self.w,x0)
        q[1] = np.dot(self.w,x1)
        q[2] = np.dot(self.w,x2)
        q[3] = np.dot(self.w,x3)

        # tie breaking
        i_max = np.random.choice(np.flatnonzero(q == q.max()))

        self.Q[state[0],state[1],state[2],0] = q[0]
        self.Q[state[0],state[1],state[2],1] = q[1]
        self.Q[state[0],state[1],state[2],2] = q[2]
        self.Q[state[0],state[1],state[2],3] = q[3]

        return i_max, 0

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

        delta = reward
        feature = self.feature(self.last_state, self.last_action)
        for i in range(len(feature)):
            delta -= self.w[feature[i]]
            self.z[feature[i]] = 1 #   replacing

        next_action, _ = self.greedy_action(state)
        next_feature = self.feature(state, next_action)
        for i in range(len(next_feature)):
            delta += self.gamma*self.w[next_feature[i]]

        self.w += self.alpha*delta*self.z
        self.z = self.gamma*self.lmbda*self.z

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
        delta = reward
        feature = self.feature(self.last_state, self.last_action)
        for i in range(len(feature)):
            delta -= self.w[feature[i]]

            #   replacing
            self.z[feature[i]] = 1

        self.w += self.alpha * delta * self.z

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
            # return self.V
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
