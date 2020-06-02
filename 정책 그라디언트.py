# %%
import tensorflow as tf
from tensorflow import keras
import gym
import matplotlib
import matplotlib.pyplot as plt
# %%
env = gym.make("CartPole-v1") # gym.make("환경 명") : 환경을 선언해준다. 
obs = env.reset() # 환경을 초기화해준다.
obs

# %%
arr = env.render(mode = 'rgb_array')
print(arr.shape)