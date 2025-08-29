import torch


class ExponentialMovingAverage:
    """EMA for model parameters.

    Maintains a shadow copy of parameters updated as:
        ema = decay * ema + (1 - decay) * param
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.collected = {}
        self.register(model)

    @torch.no_grad()
    def register(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            new_average = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]

