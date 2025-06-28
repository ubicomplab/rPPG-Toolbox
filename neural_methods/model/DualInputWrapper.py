import torch.nn as nn

class DualInputWrapper(nn.Module):
    """Wrapper to allow models to accept two inputs.

    The background input is currently ignored and only kept for API compatibility.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, face, background=None):
        return self.base_model(face)
