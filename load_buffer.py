import pickle

import numpy as np

with open("data/distillation_buffer_pong_2e5.pkl", "rb") as f:
    memory = pickle.load(f)

print(len(memory))
print(np.mean(memory.q_value_buf))
