import gym
from gym import logger as gymlogger
from gym.wrappers.record_video import RecordVideo
gymlogger.set_level(40) #error only
import tensorflow as tf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
from IPython.display import HTML

from IPython import display as ipythondisplay

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

"""
We need to pip install gym[atari,accept-rom-license]==0.21.0 to enable license of ROM and set up windows environment " 
Gym support is included in ale-py. Simply install the Python package using the instructions above. You can also install gym[atari] which also installs ale-py with Gym.
As of Gym v0.20 and onwards all Atari environments are provided via ale-py. We do recommend using the new v5 environments in the ALE namespace

"""

from ale_py import ALEInterface

ale = ALEInterface()

#env for Pacman 
env = gym.make('ALE/Breakout-v5')
print(env.action_space)

env.env.get_action_meanings()


"""
 A function to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data=''''''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

Q = np.zeros([210*160, env.action_space.n])
alpha = 0.618
G = 0
state = env.reset()
# counter = 0
# reward = None

for episode in range(1,101):
    done = False
    G, reward = 0,0
    state = env.reset()
    while done != True:
            if random.random() < (0.8 / (episode*.1)): #take less random steps as you learn more about the game
              action = random.randint(0,env.action_space.n-1)
            else:
              action = np.argmax(Q[state[:,:,0]])
            state2, reward, done, info = env.step(action) #2
            Q[state[:,:,0],action] += alpha * (reward + np.max(Q[state2[:,:,0]]) - Q[state[:,:,0],action]) #3
            G += reward
            state = state2   
    if episode % 5 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G))
        
        done = False
        G, reward = 0,0
        state = env.reset()
        while done != True:
            action = np.argmax(Q[state[:,:,0]])
            state2, reward, done, info = env.step(action) 
            Q[state[:,:,0],action] += alpha * (reward + np.max(Q[state2[:,:,0]]) - Q[state[:,:,0],action]) #3
            G += reward
            state = state2 
            
env.close()
show_video()