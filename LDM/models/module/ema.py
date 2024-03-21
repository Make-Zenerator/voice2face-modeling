import torch

class EMA():
    def __init__(self, model, inv_gamma=1.0, power=1.0, min_value=0.0):
        self.model = model
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.shadow_params = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] -= (1.0 - self.inv_gamma) * (self.shadow_params[name] - param.data)
                self.shadow_params[name] = torch.max(self.shadow_params[name], param.data - self.min_value)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow_params[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow_params[name].clone()