from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from photo_edit_studio.config import settings
from photo_edit_studio.models.memory import apply_inference_memory_settings
from photo_edit_studio.models.registry import ModelSpec
from photo_edit_studio.progress import GenerationProgress
from photo_edit_studio.types import GenerationRequest


class ModelAdapter(ABC):
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.pipe: Any = None
        self._lora_signature: tuple[tuple[str, float], ...] = ()

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self, request: GenerationRequest, progress: GenerationProgress | None = None
    ) -> list[Any]:
        raise NotImplementedError

    def configure_memory(self) -> None:
        apply_inference_memory_settings(self.pipe, mode=settings.memory_mode)

    def unload(self) -> None:
        self.pipe = None
        self._lora_signature = ()

    @staticmethod
    def lora_signature(request: GenerationRequest) -> tuple[tuple[str, float], ...]:
        return tuple((str(item.path), item.weight) for item in request.loras)

    def apply_loras(self, request: GenerationRequest) -> None:
        signature = self.lora_signature(request)
        if signature == self._lora_signature:
            return
        unload = getattr(self.pipe, "unload_lora_weights", None)
        if callable(unload) and self._lora_signature:
            unload()
        if not request.loras:
            self._lora_signature = ()
            return
        load = getattr(self.pipe, "load_lora_weights", None)
        set_adapters = getattr(self.pipe, "set_adapters", None)
        if not callable(load) or not callable(set_adapters):
            raise TypeError(f"{self.spec.label} does not expose Diffusers LoRA support.")
        names: list[str] = []
        weights: list[float] = []
        for item in request.loras:
            load(str(item.path), adapter_name=item.adapter_name)
            names.append(item.adapter_name)
            weights.append(item.weight)
        set_adapters(names, adapter_weights=weights)
        self._lora_signature = signature
