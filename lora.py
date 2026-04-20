import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1):
        super().__init__()
        self.original = original_layer

        self.lora_A = nn.Linear(
            original_layer.in_features, rank, bias=False,
            device=original_layer.weight.device,
            dtype=original_layer.weight.dtype
        )
        self.lora_B = nn.Linear(
            rank, original_layer.out_features, bias=False,
            device=original_layer.weight.device,
            dtype=original_layer.weight.dtype
        )

        self.scale = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale


def inject_lora(module, rank=4):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank))
        else:
            inject_lora(child, rank)


def get_lora_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.extend(list(m.lora_A.parameters()))
            params.extend(list(m.lora_B.parameters()))
    return params