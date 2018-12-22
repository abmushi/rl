"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue

from env import GridEnvironment
from env_cliff import CliffEnvironment
from env_maze import MazeEnvironment

from agent_td import TDAgent
from agent_q import QAgent
from agent_sarsa import SARSAgent
from agent_exp_sarsa import ExpectedSARSAgent
from agent_n_sarsa import NstepSARSAgent
from agent_dyna_q import DynaQAgent
from agent_dyna_q_plus import DynaQPlusAgent

import constant as constant
import numpy as np
from time import sleep

import pygame

pygame.init()
screen = pygame.display.set_mode((700, 520))
done = False

is_learning = False
count = 0
max_episode = 200
time_step = 0

# --- select environment
# environment = GridEnvironment()
# environment = CliffEnvironment()
environment = MazeEnvironment()

# --- select agent
# agent = TDAgent()
# agent = SARSAgent()
# agent = QAgent()
# agent = ExpectedSARSAgent()
# agent = NstepSARSAgent()
# agent = DynaQAgent()
agent = DynaQPlusAgent()

rlglue = RLGlue(environment, agent)

def draw_step(V,state,env_map):
    # draw Q
    for i in range(0,constant._x):
        for j in range(0,constant._y):
            rect = pygame.Rect(i*constant._unit,j*constant._unit,constant._unit,constant._unit)
            if env_map[i,j] == constant.TYPE_GOAL:
                screen.fill((255,255,0),rect)
            elif env_map[i,j] == constant.TYPE_CLIFF:
                screen.fill((50,50,53),rect)
            elif env_map[i,j] == constant.TYPE_WALL:
                screen.fill((150,150,150),rect)
            else:
                temp = V[i,j]*200
                if temp > 255:
                    temp = 255
                elif temp < 0:
                    temp = 0
                temp2 = int(255-temp)
                screen.fill((255,temp2,temp2),rect)

    # draw line
    for i in range(0,constant._x):
        pygame.draw.line(screen,(255,255,255),(i*constant._unit,0),(i*constant._unit,constant._y*constant._unit))
    for j in range(0,constant._y):
        pygame.draw.line(screen,(255,255,255),(0,j*constant._unit),(constant._x*constant._unit,j*constant._unit))

    # draw agent
    agent_rect = pygame.Rect(state[0]*constant._unit + 4, state[1]*constant._unit + 4, constant._unit-8, constant._unit-8)
    screen.fill((0,0,0),agent_rect)

def learn_step():
    global is_learning, count, max_episode
    if not is_learning:
        rlglue.rl_start()
        is_learning = True
    if is_learning:
        reward, state, action, is_terminal = rlglue.rl_step()
        if is_terminal == True:
            is_learning = False
        elif count > max_episode:
            # print('<-------- count over -------->')
            is_learning = False
            count = 0

    return rlglue.rl_agent_message('V'), rlglue.rl_agent_message('STATE'), rlglue.rl_env_message('MAP')

def run():
    global done,count

    # np.random.seed(count)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Q = rlglue.rl_agent_message('Q')
            print(Q)
            done = True

    V, state, env_map = learn_step()
    draw_step(V,state,env_map)

    pygame.display.flip()


if __name__ == "__main__":
    # main()
    rlglue.rl_init()
    while not done:
        run()
        count += 1

        if time_step % 100 == 1:
            count_episode = rlglue.rl_agent_message('COUNT')
            print('time_step: {:d}, count: {:d}'.format(time_step,count_episode))

        time_step += 1
        # print(count)
        # sleep(0.5)
