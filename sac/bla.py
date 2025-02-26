import torch

model_path = r"C:\Users\regin\Documents\Winter24_25\rl_project\sac\logs\agents\10.pth"
state = torch.load(model_path, map_location="cpu")

print("📂 Gespeicherte Keys in state_dict:")
print(state.keys())
