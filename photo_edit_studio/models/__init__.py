from __future__ import annotations

import logging
from typing import Any

from photo_edit_studio.models.base import ModelAdapter
from photo_edit_studio.models.comfy_swap import ComfyFireRedAdapter, ComfyQwen21Adapter
from photo_edit_studio.models.diffusers_adapters import (
    FluxKleinAdapter,
    Krea2Adapter,
    QwenAdapter,
    QwenAioAdapter,
)
from photo_edit_studio.models.memory import release_cuda
from photo_edit_studio.models.registry import MODEL_SPECS

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self) -> None:
        self._adapter: ModelAdapter | None = None
        self._key: str | None = None

    @property
    def status(self) -> str:
        if self._adapter is None:
            return "No model loaded"
        return f"Loaded: {MODEL_SPECS[self._key].label} (sequential CPU offload)"

    def get(self, key: str) -> ModelAdapter:
        if key not in MODEL_SPECS:
            raise KeyError(f"Unknown model: {key}")
        spec = MODEL_SPECS[key]
        if self._key == key and self._adapter is not None:
            logger.info("Model manager adapter reuse")
            return self._adapter
        logger.info("Model manager adapter selection begin")
        self.unload()
        adapter_type: type[ModelAdapter]
        if spec.loader == "qwen_aio":
            adapter_type = QwenAioAdapter
        elif spec.loader == "firered_comfy":
            adapter_type = ComfyFireRedAdapter
        elif spec.loader == "qwen21_comfy":
            adapter_type = ComfyQwen21Adapter
        elif spec.loader == "krea2":
            adapter_type = Krea2Adapter
        elif spec.loader == "qwen":
            adapter_type = QwenAdapter
        elif spec.loader == "flux":
            adapter_type = FluxKleinAdapter
        else:
            raise RuntimeError(f"Unsupported model loader: {spec.loader}")
        self._adapter = adapter_type(spec)
        self._key = key
        logger.info("Model manager adapter selection end: adapter=%s", adapter_type.__name__)
        return self._adapter

    def unload(self) -> str:
        logger.info("Model manager unload begin")
        if self._adapter is not None:
            self._adapter.unload()
        self._adapter = None
        self._key = None
        release_cuda()
        logger.info("Model manager unload end")
        return "No model loaded; CUDA cache released"


model_manager = ModelManager()

__all__: list[Any] = ["MODEL_SPECS", "ModelManager", "model_manager"]
