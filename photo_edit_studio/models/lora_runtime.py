from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def clone_inference_parameters(module: Any) -> int:
    """Replace inference tensors with ordinary Parameters PEFT can safely mutate."""
    import torch

    replaced = 0
    for name, parameter in list(module._parameters.items()):
        if parameter is not None and parameter.is_inference():
            module._parameters[name] = torch.nn.Parameter(
                parameter.detach().clone(), requires_grad=parameter.requires_grad
            )
            replaced += 1
    for child in module.children():
        replaced += clone_inference_parameters(child)
    return replaced


def _peft_layers(pipe: Any) -> list[Any]:
    root = getattr(pipe, "transformer", pipe)
    return [module for module in root.modules() if callable(getattr(module, "set_adapter", None))]


def activate_peft_adapters(pipe: Any, names: list[str], weights: list[float]) -> None:
    for layer in _peft_layers(pipe):
        layer.set_adapter(names, inference_mode=True)
        for name, weight in zip(names, weights, strict=True):
            set_scale = getattr(layer, "set_scale", None)
            if callable(set_scale):
                set_scale(name, weight)


def set_peft_adapter_scales(pipe: Any, names: Iterable[str], weights: Iterable[float]) -> None:
    pairs = list(zip(names, weights, strict=True))
    for layer in _peft_layers(pipe):
        set_scale = getattr(layer, "set_scale", None)
        if callable(set_scale):
            for name, weight in pairs:
                set_scale(name, weight)
