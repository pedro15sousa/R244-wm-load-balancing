"""
Define Controller
"""
import torch
import torch.nn as nn  
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=-1)
        out = self.fc(cat_in)
        # Apply softmax to get probability distribution over actions
        return F.softmax(out, dim=-1)
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            # Unpack the state if it's a tuple
            if isinstance(state, tuple):
                probabilities = self.forward(*state)
            else:
                probabilities = self.forward(state)
            if deterministic:
                # Choose the action with the highest probability
                return probabilities.argmax(dim=-1).item()
            else:
                # Sample an action according to the probabilities
                return torch.multinomial(probabilities, 1).item()