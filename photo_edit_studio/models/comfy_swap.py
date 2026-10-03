from __future__ import annotations

import json
import os
import time
import uuid
from copy import deepcopy
from typing import Any

import httpx
from PIL import Image

from photo_edit_studio.comfy_assets import (
    KREA_FIRST_LORA_FILE,
    QWEN21_TURBO_LORA,
    missing_firered_assets,
    missing_krea_remix_assets,
    missing_qwen21_assets,
)
from photo_edit_studio.comfy_backend import _endpoint, ensure_backend
from photo_edit_studio.comfy_workflows import (
    firered_edit_template,
    krea_reference_template,
    krea_remix_template,
    krea_text_template,
    qwen21_turbo_template,
)
from photo_edit_studio.config import settings
from photo_edit_studio.image_utils import constrain_output_size, normalize_image
from photo_edit_studio.models.base import ModelAdapter
from photo_edit_studio.models.diffusers_adapters import FRAMING_SUFFIX
from photo_edit_studio.models.memory import cuda_available, cuda_vram_gb
from photo_edit_studio.progress import GenerationProgress
from photo_edit_studio.swap import swap_profile
from photo_edit_studio.types import GenerationRequest, LoraSpec


def _api_graph(workflow: dict[str, Any]) -> dict[str, Any]:
    if not workflow or not all(isinstance(node, dict) and "class_type" in node for node in workflow.values()):
        raise ValueError(
            "ComfyUI requires an API-format workflow (node IDs mapped to class_type and inputs). "
            "The upstream BFS JSON files are UI-format and cannot be queued directly; "
            "export the workflow using ComfyUI > Save (API Format)."
        )
    return workflow


def _local_comfy_url(url: str) -> str:
    _endpoint(url)
    return url.rstrip("/")


def apply_mandatory_krea_lora(
    graph: dict[str, Any], request: GenerationRequest
) -> dict[str, Any]:
    """Prepend the mandatory adapter to every path from the Krea UNET loader."""
    path = settings.comfy_dir / "models" / "loras" / KREA_FIRST_LORA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Mandatory Krea LoRA is missing: {path}")
    weight = float(request.krea_first_lora_weight)
    if not 0 < weight <= 2:
        raise ValueError("Mandatory Krea LoRA weight must be greater than 0 and at most 2.")
    source = graph.get("55")
    if not isinstance(source, dict) or source.get("class_type") != "UNETLoader":
        raise ValueError("Krea workflow needs UNETLoader node 55 for the mandatory LoRA.")
    consumers = [
        inputs for node_id, node in graph.items() if node_id != "55"
        for inputs in [node.get("inputs", {})]
        if inputs.get("model") == ["55", 0]
    ]
    if not consumers:
        raise ValueError("Krea UNETLoader is not connected to a model consumer.")
    for node in graph.values():
        if (node.get("class_type") == "LoraLoaderModelOnly"
                and str(node.get("inputs", {}).get("lora_name", "")).replace("\\", "/").rsplit("/", 1)[-1].casefold()
                == KREA_FIRST_LORA_FILE.casefold()):
            raise ValueError("Mandatory Krea LoRA is already in the workflow; remove the duplicate node.")
    node_id = str(max((int(key) for key in graph if key.isdecimal()), default=0) + 1)
    graph[node_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["55", 0], "lora_name": KREA_FIRST_LORA_FILE, "strength_model": weight},
    }
    for inputs in consumers:
        inputs["model"] = [node_id, 0]
    return graph


def apply_krea_loras(
    graph: dict[str, Any], anchor_id: str, loras: list[LoraSpec], *, adjust_base_weight: bool = True
) -> dict[str, Any]:
    """Adjust a selected base LoRA and chain other Krea LoRAs after it."""
    if not loras:
        return graph
    anchor = graph.get(anchor_id)
    if not isinstance(anchor, dict) or anchor.get("class_type") != "LoraLoaderModelOnly":
        raise ValueError(f"Krea workflow needs a base LoraLoaderModelOnly at node {anchor_id}.")
    consumers = [
        inputs for node_id, node in graph.items() if node_id != anchor_id
        for inputs in [node.get("inputs", {})]
        if inputs.get("model") == [anchor_id, 0]
    ]
    if not consumers:
        raise ValueError(f"Krea base LoRA node {anchor_id} is not connected to the model.")
    base_name = anchor["inputs"].get("lora_name", "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    family_dir = (settings.lora_dir / "krea2").resolve()
    names: set[str] = set()
    additional: list[LoraSpec] = []
    selected_base_weight: float | None = None
    for lora in loras:
        if lora.name.casefold() == KREA_FIRST_LORA_FILE.casefold():
            raise ValueError("The mandatory Krea LoRA has its own weight control; do not select it in an optional slot.")
        path = lora.path.resolve()
        if path.parent != family_dir or path.name != lora.name or path.suffix.lower() != ".safetensors":
            raise ValueError("ComfyUI accepts only Krea2 .safetensors adapters in models/loras/krea2/.")
        if not path.is_file():
            raise FileNotFoundError(f"Selected Krea LoRA is missing: {path}")
        if path.name.casefold() in names:
            raise ValueError(f"Krea LoRA {path.name} is selected more than once.")
        names.add(path.name.casefold())
        if not 0 <= lora.weight <= 2:
            raise ValueError("Krea LoRA weight must be between 0 and 2.")
        if path.name.casefold() == base_name.casefold():
            selected_base_weight = float(lora.weight)
        else:
            additional.append(lora)
    if adjust_base_weight and selected_base_weight is not None:
        anchor["inputs"]["strength_model"] = selected_base_weight
    previous = anchor_id
    next_id = max((int(key) for key in graph if key.isdecimal()), default=0) + 1
    for lora in additional:
        current = str(next_id)
        graph[current] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [previous, 0], "lora_name": lora.name,
                "strength_model": float(lora.weight),
            },
        }
        previous = current
        next_id += 1
    for inputs in consumers:
        inputs["model"] = [previous, 0]
    return graph


def validate_krea_remix_request(request: GenerationRequest) -> None:
    """Reject unsupported inputs before truncation, upload, or model allocation."""
    if request.model_key != "krea-2-turbo" or request.workflow != "krea-remix":
        raise ValueError("krea-remix supports only model_key='krea-2-turbo'.")
    if len(request.images) != 1:
        raise ValueError("Krea remix requires exactly one uploaded composed canvas; no second image.")
    if request.mask is not None:
        raise ValueError("Krea remix does not support a mask.")
    if request.count != 1:
        raise ValueError("Krea remix supports a single output only (count=1).")
    if request.loras:
        raise ValueError("Krea remix does not support extra LoRAs; Remix is first-pass only.")
    if request.size_multiplier != 1:
        raise ValueError("Krea remix supports size_multiplier=1 only; no upscale.")
    if not isinstance(request.steps, int) or not 3 <= request.steps <= 40:
        raise ValueError("Krea remix steps must be between 3 and 40 (default 9).")
    if request.guidance != 1:
        raise ValueError("Krea remix requires guidance=1.")
    if request.negative_prompt.strip():
        raise ValueError("Krea remix does not support a negative prompt at guidance=1.")
    if not 0 < request.krea_first_lora_weight <= 2:
        raise ValueError("Mandatory Krea LoRA weight must be greater than 0 and at most 2.")


def krea_remix_size(request: GenerationRequest) -> tuple[int, int]:
    return constrain_output_size(
        normalize_image(request.images[0], max_side=1024).size,
        max_side=min(1024, settings.max_output_side),
        max_pixels=min(1024 * 1024, settings.max_output_pixels),
    )


def configure_krea_remix_graph(
    graph: dict[str, Any], request: GenerationRequest, image_names: tuple[str, ...]
) -> dict[str, Any]:
    validate_krea_remix_request(request)
    if len(image_names) != 1:
        raise ValueError("Krea remix needs exactly one uploaded image name.")
    width, height = krea_remix_size(request)
    graph["72"]["inputs"]["image"] = image_names[0]
    graph["73"]["inputs"].update(width=width, height=height)
    graph["84"]["inputs"]["prompt"] = request.prompt
    graph["86"]["inputs"]["text"] = request.prompt
    for node_id in ("53", "54"):
        graph[node_id]["inputs"].update(noise_seed=request.seed, steps=request.steps, cfg=1.0)
    graph["53"]["inputs"]["end_at_step"] = request.steps - 1
    graph["54"]["inputs"].update(start_at_step=request.steps - 1, end_at_step=request.steps)
    graph["29"]["inputs"]["filename_prefix"] = f"photo_edit_krea_remix_{uuid.uuid4().hex}"
    return apply_mandatory_krea_lora(graph, request)


def preflight_krea_remix_graph(graph: dict[str, Any], registered: dict[str, Any]) -> None:
    required = {node["class_type"] for node in graph.values()}
    missing = required - set(registered)
    if missing:
        raise RuntimeError(
            f"Krea remix is missing ComfyUI nodes: {', '.join(sorted(missing))}. "
            "Run scripts/setup-comfy.ps1 to install https://github.com/ostris/ComfyUI-Krea2-Ostris-Edit "
            "and restart ComfyUI. Identity-edit nodes cannot substitute for Ostris nodes."
        )
    for node in graph.values():
        schema = registered[node["class_type"]].get("input", {})
        inputs = {**schema.get("required", {}), **schema.get("optional", {})}
        for field, value in node["inputs"].items():
            if field not in inputs:
                raise RuntimeError(f"Krea remix node {node['class_type']} lacks input {field}; update nodes and restart ComfyUI.")
            choices = inputs[field][0]
            if node["class_type"] == "LoadImage" and field == "image":
                # Upload occurs AFTER preflight; its new filename is not listed yet.
                continue
            if isinstance(choices, list) and not isinstance(value, list):
                matches = [choice for choice in choices if str(choice).replace('\\', '/') == str(value).replace('\\', '/')]
                if not matches:
                    raise RuntimeError(f"Krea remix {node['class_type']} has not registered {field}={value}; check assets/update ComfyUI and restart.")
                node["inputs"][field] = matches[0]


def configure_krea_text_graph(
    graph: dict[str, Any], request: GenerationRequest
) -> dict[str, Any]:
    if request.model_key != "krea-2-turbo" or request.images:
        raise ValueError("Krea text workflow requires a Krea model and no input images.")
    graph["84"]["inputs"]["text"] = request.prompt
    graph["85"]["inputs"]["text"] = request.negative_prompt
    graph["53"]["inputs"].update(seed=request.seed, steps=request.steps, cfg=request.guidance)
    graph["82"]["inputs"].update(width=request.width, height=request.height, batch_size=1)
    graph["29"]["inputs"]["filename_prefix"] = f"photo_edit_krea_text_{uuid.uuid4().hex}"
    apply_mandatory_krea_lora(graph, request)
    first_id = graph["53"]["inputs"]["model"][0]
    return apply_krea_loras(graph, first_id, request.loras)


def configure_krea_graph(
    graph: dict[str, Any], request: GenerationRequest, image_names: tuple[str, str]
) -> dict[str, Any]:
    """Fill the locally exported BFS API graph; fail if required nodes are absent."""
    profile = swap_profile(request.model_key, request.swap_kind or "Head")

    def node(node_id: str, expected: str) -> dict[str, Any]:
        entry = graph.get(node_id)
        if not isinstance(entry, dict) or entry.get("class_type") != expected:
            raise ValueError(f"ComfyUI workflow must contain node {node_id} ({expected}).")
        return entry["inputs"]

    body_id, reference_id = ("72", "90") if request.swap_kind == "Head" else ("72", "139")
    if "71" in graph and "79" in graph and "84" in graph:
        configure_krea_reference_graph(graph, request, image_names, apply_selected_loras=False)
        lora = node("71", "LoraLoaderModelOnly")
        lora["lora_name"] = profile.filename
        if not request.loras:
            raise ValueError("A BFS LoRA is required for Krea swap.")
        lora["strength_model"] = request.loras[0].weight
        node("84", "Krea2EditGroundedEncode")["prompt"] = (
            f"{profile.trigger} {request.prompt}".strip()
        )
        apply_krea_loras(graph, "71", request.loras[1:], adjust_base_weight=False)
        return apply_mandatory_krea_lora(graph, request)
    node(body_id, "LoadImage")["image"] = image_names[0]
    node(reference_id, "LoadImage")["image"] = image_names[1]
    lora_id = "111" if request.swap_kind == "Head" else "127"
    lora = node(lora_id, "LoraLoaderModelOnly")
    lora["lora_name"] = profile.filename
    if not request.loras:
        raise ValueError("A BFS LoRA is required for Krea swap.")
    lora["strength_model"] = request.loras[0].weight
    positive = node("119", "Krea2EditGroundedEncode")
    positive["prompt"] = f"{profile.trigger} {request.prompt}".strip()
    sampler = node("53", "KSampler")
    sampler["seed"] = request.seed
    sampler["steps"] = request.steps
    sampler["cfg"] = request.guidance
    node("29", "SaveImage")["filename_prefix"] = f"photo_edit_swap_{uuid.uuid4().hex}"
    if request.swap_kind == "Body":
        primitive = graph.get("145")
        if primitive and primitive.get("class_type") == "PrimitiveNode":
            primitive["inputs"]["value"] = positive["prompt"]
    apply_krea_loras(graph, lora_id, request.loras[1:], adjust_base_weight=False)
    return apply_mandatory_krea_lora(graph, request)


def configure_krea_reference_graph(
    graph: dict[str, Any],
    request: GenerationRequest,
    image_names: tuple[str, ...],
    *,
    apply_selected_loras: bool = True,
) -> dict[str, Any]:
    """Configure the Krea2Edit two-reference graph; reject disconnected/ignored inputs."""
    if request.model_key != "krea-2-turbo" or not 1 <= len(image_names) <= 2:
        raise ValueError("Krea reference editing needs a source and at most one reference.")
    source = graph.get("72")
    positive = graph.get("84")
    negative = graph.get("85")
    patch = graph.get("79")
    sampler = graph.get("53")
    saver = graph.get("29")
    if any(
        not isinstance(entry, dict) or entry.get("class_type") != class_type
        for entry, class_type in (
            (source, "LoadImage"), (positive, "Krea2EditGroundedEncode"),
            (negative, "Krea2EditGroundedEncode"), (patch, "Krea2EditModelPatch"),
            (sampler, "KSampler"), (saver, "SaveImage"),
        )
    ):
        raise ValueError("Krea reference API workflow is missing required Krea2Edit nodes.")
    source["inputs"]["image"] = image_names[0]
    if len(image_names) == 1:
        for entry, field in ((positive, "image_b"), (negative, "image_b"),
                             (patch, "source_latent_b"), (patch, "source_image_b")):
            entry.get("inputs", {}).pop(field, None)
        for node_id in ("90", "92"):
            graph.pop(node_id, None)
    else:
        second = graph.get("90")
        if not isinstance(second, dict) or second.get("class_type") != "LoadImage":
            raise ValueError("Krea reference API workflow needs a second LoadImage (node 90).")
        # ComfyUI API exports omit bypassed nodes. Require actual graph links into both
        # image-grounding and latent conditioning instead of silently ignoring image 2.
        for entry, field in (
            (positive, "image_b"), (negative, "image_b"),
            (patch, "source_latent_b"), (patch, "source_image_b"),
        ):
            if not isinstance(entry.get("inputs", {}).get(field), list):
                raise TypeError(
                    "The second Krea reference is disconnected. Enable group 2 in ComfyUI "
                    "and export a two-image API-format workflow."
                )
        if positive["inputs"]["image_b"][0] != "90" or negative["inputs"]["image_b"][0] != "90":
            raise ValueError("Krea second reference must be connected to LoadImage node 90.")
        if patch["inputs"]["source_image_b"][0] != "90":
            raise ValueError("Krea second reference pixel path must come from LoadImage node 90.")
        latent_b = patch["inputs"]["source_latent_b"][0]
        encoder = graph.get(str(latent_b))
        if not isinstance(encoder, dict) or encoder.get("class_type") != "VAEEncode" or encoder.get("inputs", {}).get("pixels", [None])[0] != "90":
            raise ValueError("Krea second reference must be VAE-encoded from LoadImage node 90.")
        second["inputs"]["image"] = image_names[1]
    positive["inputs"]["prompt"] = request.prompt
    negative["inputs"]["prompt"] = request.negative_prompt
    sampler["inputs"].update(seed=request.seed, steps=request.steps, cfg=request.guidance)
    latent = graph.get("82")
    if not isinstance(latent, dict) or latent.get("class_type") != "EmptySD3LatentImage":
        raise ValueError("Krea reference workflow needs output latent node 82.")
    latent["inputs"].update(width=request.width, height=request.height, batch_size=1)
    saver["inputs"]["filename_prefix"] = f"photo_edit_krea_ref_{uuid.uuid4().hex}"
    lora = graph.get("71")
    if not isinstance(lora, dict) or lora.get("class_type") != "LoraLoaderModelOnly":
        raise ValueError("Krea reference workflow needs the identity-edit LoRA at node 71.")
    lora["inputs"]["lora_name"] = settings.comfy_reference_lora
    if apply_selected_loras:
        apply_krea_loras(graph, "71", request.loras)
        apply_mandatory_krea_lora(graph, request)
    return graph


def configure_firered_graph(
    graph: dict[str, Any], request: GenerationRequest, image_names: tuple[str, ...]
) -> dict[str, Any]:
    if request.model_key != "firered-1.1" or request.workflow != "standard":
        raise ValueError("FireRed GGUF supports only image edit and combine workflows.")
    if not 1 <= len(image_names) <= 3 or len(image_names) != len(request.images):
        raise ValueError("FireRed needs one to three uploaded images.")
    if not 1 <= request.steps <= 40 or not 0 <= request.true_cfg <= 10:
        raise ValueError("FireRed steps or CFG are outside supported controls.")
    if request.width % 16 or request.height % 16:
        raise ValueError("FireRed output width and height must be divisible by 16.")
    if request.loras:
        raise ValueError(
            "FireRed's Lightning v1.2 LoRA is loaded automatically; additional LoRAs are not "
            "validated for this quantized 12 GB workflow."
        )
    graph["7"]["inputs"]["image"] = image_names[0]
    if request.compose:
        graph["12"]["inputs"]["latent_image"] = ["14", 0]
        graph["14"] = {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": request.width, "height": request.height, "batch_size": 1,
        }}
        graph.pop("9")
    else:
        graph["14"]["inputs"].update(width=request.width, height=request.height)
    for index, name in enumerate(image_names[1:], start=2):
        node_id = str(18 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        for encoder in ("10", "11"):
            graph[encoder]["inputs"][f"image{index}"] = [node_id, 0]
    graph["10"]["inputs"]["prompt"] = request.prompt + FRAMING_SUFFIX
    graph["11"]["inputs"]["prompt"] = request.negative_prompt
    graph["12"]["inputs"].update(seed=request.seed, steps=request.steps, cfg=request.true_cfg)
    graph["29"]["inputs"]["filename_prefix"] = f"photo_edit_firered_{uuid.uuid4().hex}"
    return graph


def configure_qwen21_graph(
    graph: dict[str, Any], request: GenerationRequest, image_names: tuple[str, ...]
) -> dict[str, Any]:
    """Connect every uploaded reference to the Qwen2.1 conditioner, with output size from the request."""
    if request.model_key != "qwen-2.1-turbo" or request.workflow not in {"standard", "text", "swap"}:
        raise ValueError("Qwen Image 2.1 Turbo supports edit, combine, text creation and swap only.")
    text_only = request.workflow == "text"
    if len(image_names) != len(request.images) or (image_names and text_only) or (not text_only and not 1 <= len(image_names) <= 3):
        raise ValueError("Qwen Image 2.1 Turbo requires 1–3 images for editing, or none for text creation.")
    if request.workflow == "swap" and (len(image_names) != 2 or request.swap_kind not in {"Head", "Body"}):
        raise ValueError("Qwen Image 2.1 swap requires a target and donor image and Head or Body selection.")
    if request.steps != 6 or request.true_cfg != 1.0 or request.guidance != 1.0:
        raise ValueError("Viggle Turbo r256 requires exactly 6 steps, CFG 1 and the fixed six-step sigma schedule.")
    if request.negative_prompt.strip():
        raise ValueError("Viggle Turbo uses no negative prompt.")
    if request.width % 32 or request.height % 32:
        raise ValueError("Qwen Image 2.1 output dimensions must be divisible by 32.")
    graph["5"]["inputs"]["prompt"] = request.prompt if text_only else request.prompt + FRAMING_SUFFIX
    graph["6"]["inputs"].update(width=request.width, height=request.height)
    graph["7"]["inputs"]["noise_seed"] = request.seed
    graph["29"]["inputs"]["filename_prefix"] = f"photo_edit_qwen21_{uuid.uuid4().hex}"
    if len(request.loras) > 5:
        raise ValueError("Qwen 2.1 supports at most five optional LoRAs after Viggle Turbo.")
    lora_dir = (settings.lora_dir / "qwen21").resolve()
    seen: set[str] = set()
    previous = "2"
    for index, lora in enumerate(request.loras, start=1):
        path = lora.path.resolve()
        if (path.parent != lora_dir or path.name != lora.name
                or path.suffix.lower() != ".safetensors"
                or path.name.casefold() == QWEN21_TURBO_LORA.casefold()):
            raise ValueError("Qwen 2.1 only accepts optional .safetensors LoRAs from models/loras/qwen21/.")
        if not path.is_file():
            raise FileNotFoundError(f"Selected Qwen 2.1 LoRA is missing: {path}")
        if path.name.casefold() in seen:
            raise ValueError(f"Qwen 2.1 LoRA {path.name} was selected more than once.")
        seen.add(path.name.casefold())
        if not 0 <= lora.weight <= 2:
            raise ValueError("Qwen 2.1 optional LoRA weight must be between 0 and 2.")
        current = str(39 + index)
        graph[current] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": [previous, 0], "lora_name": os.path.join("qwen21", path.name),
            "strength_model": float(lora.weight),
        }}
        previous = current
    graph["8"]["inputs"]["model"] = [previous, 0]
    for index, name in enumerate(image_names, start=1):
        node_id = str(30 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph["5"]["inputs"][f"images.image_{index}"] = [node_id, 0]
    # The sampler and Viggle sigma scheduler must always see the SAME latent.
    # Use the UI's explicit canvas size for edits, combine, text, and swaps.
    return graph


def resolve_qwen21_lora_names(graph: dict[str, Any], choices: list[str]) -> None:
    """Use exactly the LoRA names returned by ComfyUI's platform-specific list."""
    for node in graph.values():
        if node["class_type"] != "LoraLoaderModelOnly":
            continue
        requested = node["inputs"]["lora_name"]
        matches = [name for name in choices if name.replace("\\", "/") == requested.replace("\\", "/")]
        if not matches:
            raise RuntimeError(
                f"ComfyUI has not registered the selected Qwen 2.1 LoRA: {requested}. "
                "Restart the project-managed ComfyUI backend after adding LoRA files."
            )
        node["inputs"]["lora_name"] = next((name for name in matches if name == requested), matches[0])


class ComfyQwen21Adapter(ModelAdapter):
    def load(self) -> None:
        raise RuntimeError("Qwen Image 2.1 Turbo runs through the project-managed ComfyUI API.")

    def generate(
        self, request: GenerationRequest, progress: GenerationProgress | None = None
    ) -> list[Image.Image]:
        if not cuda_available() or cuda_vram_gb() + 0.5 < self.spec.minimum_vram_gb:
            raise RuntimeError("Qwen Image 2.1 Turbo requires a CUDA GPU with about 12 GB VRAM and CPU offload.")
        configure_qwen21_graph(
            qwen21_turbo_template(), request,
            tuple(f"image_{i}.png" for i in range(len(request.images))),
        )
        missing = missing_qwen21_assets(settings.comfy_dir)
        if missing:
            raise FileNotFoundError(
                "Qwen Image 2.1 INT8 weights, Viggle r256 LoRA or custom node missing. "
                "Run scripts/setup-comfy.ps1 and scripts/download_models.py qwen-2.1-turbo. "
                f"Missing: {', '.join(missing)}"
            )
        url = _local_comfy_url(settings.comfy_url)
        ensure_backend(model_key="qwen-2.1-turbo")
        results: list[Image.Image] = []
        from io import BytesIO

        with httpx.Client(base_url=url, timeout=60.0, trust_env=False) as client:
            try:
                client.get("/system_stats").raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Cannot connect to local ComfyUI at {url}.") from exc
            required_nodes = {"TextEncodeQwenImage21", "ViggleTurboLora", "ViggleTurboSigmas"}
            try:
                available = client.get("/object_info")
                available.raise_for_status()
                registered = available.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise RuntimeError("Cannot verify Qwen 2.1 and Viggle custom nodes in ComfyUI.") from exc
            missing_nodes = required_nodes - set(registered)
            if missing_nodes:
                raise RuntimeError(
                    f"ComfyUI is missing Qwen 2.1/Viggle nodes: {', '.join(sorted(missing_nodes))}. "
                    "Update ComfyUI and restart the project-managed backend."
                )
            # ComfyUI validates lora_name against its platform-specific filename
            # list before queueing. On Windows it contains backslashes, not '/'.
            lora_options = registered.get("LoraLoaderModelOnly", {}).get("input", {}).get(
                "required", {}
            ).get("lora_name", [[]])[0]
            graph_loras = configure_qwen21_graph(
                qwen21_turbo_template(), request,
                tuple(f"image_{i}.png" for i in range(len(request.images))),
            )
            resolve_qwen21_lora_names(graph_loras, lora_options)
            names = []
            for image in request.images:
                data = BytesIO()
                normalize_image(image, max_side=1024).save(data, format="PNG")
                response = client.post(
                    "/upload/image",
                    files={"image": (f"qwen21_{uuid.uuid4().hex}.png", data.getvalue(), "image/png")},
                    data={"type": "input", "overwrite": "false"},
                )
                response.raise_for_status()
                names.append(response.json()["name"])
            for index in range(request.count):
                graph = configure_qwen21_graph(qwen21_turbo_template(), request, tuple(names))
                resolve_qwen21_lora_names(graph, lora_options)
                graph["7"]["inputs"]["noise_seed"] = request.seed + index
                response = client.post("/prompt", json={"prompt": graph})
                if response.status_code >= 400:
                    raise RuntimeError(f"ComfyUI rejected the Qwen 2.1 Turbo graph: {response.text[:500]}")
                prompt_id = response.json()["prompt_id"]
                deadline = time.monotonic() + settings.comfy_timeout_seconds
                while time.monotonic() < deadline:
                    history = client.get(f"/history/{prompt_id}")
                    history.raise_for_status()
                    job = history.json().get(prompt_id)
                    if job:
                        if job.get("status", {}).get("status_str") == "error":
                            raise RuntimeError(f"Qwen 2.1 Turbo failed in ComfyUI; see {settings.output_dir / 'comfyui.log'}.")
                        images = job.get("outputs", {}).get("29", {}).get("images", [])
                        if images:
                            for item in images:
                                image_response = client.get("/view", params={
                                    "filename": item["filename"],
                                    "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output"),
                                })
                                image_response.raise_for_status()
                                with Image.open(BytesIO(image_response.content)) as image:
                                    results.append(image.convert("RGB"))
                            break
                    if progress is not None:
                        progress.update(None, "Qwen 2.1 Turbo running in low-VRAM ComfyUI · waiting for output")
                    time.sleep(1)
                else:
                    raise TimeoutError(f"Qwen 2.1 Turbo did not finish within {settings.comfy_timeout_seconds} seconds.")
        return results


class ComfyFireRedAdapter(ModelAdapter):
    def load(self) -> None:
        raise RuntimeError("FireRed uses a local ComfyUI GGUF workflow, not a Diffusers pipeline.")

    def generate(
        self, request: GenerationRequest, progress: GenerationProgress | None = None
    ) -> list[Image.Image]:
        if not 1 <= len(request.images) <= 3:
            raise ValueError("FireRed requires one to three images.")
        configure_firered_graph(
            firered_edit_template(), request, tuple(f"image_{i}.png" for i in range(len(request.images)))
        )
        missing = missing_firered_assets(settings.comfy_dir)
        if missing:
            raise FileNotFoundError(
                "FireRed GGUF Q4_K_M, FP8 vision encoder, VAE, Lightning v1.2 or ComfyUI-GGUF "
                "is missing. Run scripts/setup-comfy.ps1 and scripts/download_models.py "
                f"firered-1.1. Missing: {', '.join(missing)}"
            )
        url = _local_comfy_url(settings.comfy_url)
        ensure_backend(model_key="firered-1.1")
        results: list[Image.Image] = []
        with httpx.Client(base_url=url, timeout=60.0, trust_env=False) as client:
            try:
                client.get("/system_stats").raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Cannot connect to local ComfyUI at {url}.") from exc
            names = []
            from io import BytesIO

            for index, image in enumerate(request.images):
                data = BytesIO()
                normalize_image(image, max_side=1024).save(data, format="PNG")
                response = client.post(
                    "/upload/image",
                    files={"image": (f"firered_{uuid.uuid4().hex}.png", data.getvalue(), "image/png")},
                    data={"type": "input", "overwrite": "false"},
                )
                response.raise_for_status()
                names.append(response.json()["name"])
            for index in range(request.count):
                graph = configure_firered_graph(firered_edit_template(), request, tuple(names))
                graph["12"]["inputs"]["seed"] = request.seed + index
                queued = client.post("/prompt", json={"prompt": graph})
                if queued.status_code >= 400:
                    raise RuntimeError(
                        "ComfyUI rejected FireRed GGUF/Lightning. Confirm ComfyUI-GGUF and "
                        f"the four official weights are installed. API response: {queued.text[:500]}"
                    )
                prompt_id = queued.json()["prompt_id"]
                deadline = time.monotonic() + settings.comfy_timeout_seconds
                while time.monotonic() < deadline:
                    history = client.get(f"/history/{prompt_id}")
                    history.raise_for_status()
                    job = history.json().get(prompt_id)
                    if job:
                        if job.get("status", {}).get("status_str") == "error":
                            raise RuntimeError("ComfyUI FireRed failed; inspect the local ComfyUI log.")
                        output = job.get("outputs", {}).get("29", {}).get("images", [])
                        if output:
                            for item in output:
                                response = client.get("/view", params={
                                    "filename": item["filename"], "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output"),
                                })
                                response.raise_for_status()
                                with Image.open(BytesIO(response.content)) as image:
                                    results.append(image.convert("RGB"))
                            break
                    if progress is not None:
                        progress.update(None, "FireRed GGUF + Lightning running · CPU offload may be slow")
                    time.sleep(1)
                else:
                    raise TimeoutError(f"FireRed did not finish within {settings.comfy_timeout_seconds} seconds.")
        return results


class ComfyKreaSwapAdapter(ModelAdapter):
    def load(self) -> None:
        raise RuntimeError("Krea swapping uses a local ComfyUI API workflow, not a Diffusers pipeline.")

    def generate(
        self, request: GenerationRequest, progress: GenerationProgress | None = None
    ) -> list[Image.Image]:
        if request.workflow == "krea-remix":
            validate_krea_remix_request(request)
            missing = missing_krea_remix_assets(settings.comfy_dir)
            if missing:
                raise FileNotFoundError(
                    "Krea remix assets missing: " + ", ".join(missing) +
                    ". Place the exact Remix adapter in vendor/ComfyUI/models/loras; "
                    "identity-edit is not a substitute. No weights are downloaded automatically."
                )
            graph = configure_krea_remix_graph(krea_remix_template(), request, ("source.png",))
            workflow_path = None
        elif request.workflow == "krea-text":
            if request.images:
                raise ValueError("Krea text creation does not accept reference images.")
            graph = krea_text_template()
            workflow_path = None
        elif request.workflow == "krea-reference":
            if not 1 <= len(request.images) <= 2:
                raise ValueError("Krea reference edit needs a source and at most one reference.")
            workflow_path = settings.comfy_reference_workflow
        else:
            if request.swap_kind not in {"Head", "Body"} or len(request.images) != 2:
                raise ValueError("Krea swap needs two images and a Head or Body selection.")
            workflow_path = (
                settings.comfy_head_workflow if request.swap_kind == "Head" else settings.comfy_body_workflow
            )
        if workflow_path is not None and workflow_path.is_file():
            graph = _api_graph(json.loads(workflow_path.read_text(encoding="utf-8")))
        elif workflow_path is not None:
            graph = krea_reference_template()
        graph = deepcopy(graph)
        url = _local_comfy_url(settings.comfy_url)
        if request.workflow == "krea-remix":
            ensure_backend(workflow="krea-remix")
        else:
            ensure_backend()
        with httpx.Client(base_url=url, timeout=60.0, trust_env=False) as client:
            try:
                ready = client.get("/system_stats")
                ready.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Cannot connect to local ComfyUI at {url}. Start ComfyUI and follow "
                    "docs/KREA_REFERENCES.md before using Krea reference editing."
                ) from exc
            if request.workflow == "krea-remix":
                try:
                    response = client.get("/object_info")
                    response.raise_for_status()
                    registered = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    raise RuntimeError("Cannot verify Krea remix Ostris nodes and registered assets.") from exc
                preflight_krea_remix_graph(graph, registered)
            names = []
            for index, image in enumerate(request.images):
                if progress is not None:
                    progress.update(0.12, f"Sending source {index + 1}/{len(request.images)} to local ComfyUI")
                from io import BytesIO

                content = BytesIO()
                normalize_image(image, max_side=1024).save(content, format="PNG")
                response = client.post(
                    "/upload/image",
                    files={"image": (f"swap_{uuid.uuid4().hex}.png", content.getvalue(), "image/png")},
                    data={"type": "input", "overwrite": "false"},
                )
                response.raise_for_status()
                names.append(response.json()["name"])
            if request.workflow == "krea-remix":
                # Already configured and preflighted; change only the uploaded source.
                graph["72"]["inputs"]["image"] = names[0]
                payload = graph
            elif request.workflow == "krea-text":
                payload = configure_krea_text_graph(graph, request)
            elif request.workflow == "krea-reference":
                payload = configure_krea_reference_graph(graph, request, tuple(names))
            else:
                payload = configure_krea_graph(graph, request, (names[0], names[1]))
            queued = client.post("/prompt", json={"prompt": payload})
            if queued.status_code >= 400:
                if request.workflow == "krea-remix":
                    raise RuntimeError(f"ComfyUI rejected the Ostris Krea remix graph: {queued.text[:500]}")
                raise RuntimeError(
                    "ComfyUI rejected the swap workflow. Check that ComfyUI-Krea2Edit, "
                    "the base model, text encoder, VAE, and BFS LoRA are installed. "
                    f"API response: {queued.text[:500]}"
                )
            prompt_id = queued.json()["prompt_id"]
            deadline = time.monotonic() + settings.comfy_timeout_seconds
            while time.monotonic() < deadline:
                history = client.get(f"/history/{prompt_id}")
                history.raise_for_status()
                job = history.json().get(prompt_id)
                if job:
                    if job.get("status", {}).get("status_str") == "error":
                        raise RuntimeError("ComfyUI reported an error; inspect its local console for details.")
                    outputs = job.get("outputs", {}).get("29", {}).get("images", [])
                    if outputs:
                        if request.workflow == "krea-remix" and len(outputs) != 1:
                            raise RuntimeError("Krea remix SaveImage 29 must return exactly one generated output.")
                        results = []
                        for item in outputs:
                            image_response = client.get(
                                "/view",
                                params={"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")},
                            )
                            image_response.raise_for_status()
                            from io import BytesIO

                            with Image.open(BytesIO(image_response.content)) as image:
                                results.append(image.convert("RGB"))
                        return results
                if progress is not None:
                    progress.update(None, f"Krea {request.workflow} running in local ComfyUI · waiting for output")
                time.sleep(1)
        raise TimeoutError(f"ComfyUI did not finish within {settings.comfy_timeout_seconds} seconds.")