import pickle

with open("data/distillation_buffer.pkl", "rb") as f:
    memory = pickle.load(f)

print(memory)
