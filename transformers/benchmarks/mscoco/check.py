import torch
import json

with open('mscoco_references_emb_with_prompt.json') as f:
    emb = json.load(f)
print(torch.tensor(emb).shape)