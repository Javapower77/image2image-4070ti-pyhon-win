"""Isolated full-BF16 character sheets from the user's hash-pinned archive.

Reference adaptation: aspect-preserved <=1536 upload replaces ImageResizeKJv2
total_pixels preprocessing; the shared image goes directly to both encoders.
Auto captions use explicitly greedy TextGenerate, not unverified publisher
sampled settings. Native 3:2 canvases bypass the studio's global edit budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
import zipfile
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from photo_edit_studio.config import settings
from photo_edit_studio.dlss import add_dlss_nodes, validate_dlss_request
from photo_edit_studio.image_utils import normalize_image
from photo_edit_studio.models.base import ModelAdapter
from photo_edit_studio.models.memory import cuda_available, cuda_vram_gb
from photo_edit_studio.progress import GenerationProgress
from photo_edit_studio.types import GenerationRequest, validate_optional_lora_weight

QWEN_CHARACTER_SHEET_KEY = "qwen-2.1-sheet"
QWEN_CHARACTER_SHEET_WORKFLOW = "qwen-character-sheet"
QWEN_CHARACTER_SHEET_SHA256 = "c9a760f237ee6044aa527949cca4ed0986539cbebda94002a7631710e06f2347"
QWEN_CHARACTER_SHEET_FILES = (
    "diffusion_models/qwen_image_2.1_bf16.safetensors",
    "text_encoders/qwen3vl_8b_bf16.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
)
QWEN_CHARACTER_SHEET_PE = "text_encoders/qwen3.5_4b_int8_convrot.safetensors"
QWEN_CHARACTER_SHEET_PROFILES = {
    "Production": ("Character_Sheet_Production.json", "33b66f45-313c-482b-9520-f63fa441a34b", 592),
    "Simple": ("Character_Sheet_Simple.json", "4c425bd9-fbac-47a7-be3b-5041bb1b186f", 600),
}
QWEN_CHARACTER_SHEET_ASSET_HINT = (
    "Supply the exact user-owned archive and full BF16 assets manually; official weights: "
    "https://huggingface.co/Comfy-Org/Qwen-Image-2.1. Auto additionally needs "
    "qwen3.5_4b_int8_convrot.safetensors. Update ComfyUI for QwenImage21Cache, "
    "TextEncodeQwenImage21 and TextGenerate; install the publisher's VAEDeGrid pack "
    "and restart. KJ ImageResizeKJv2 is replaced by local aspect-preserved normalization; "
    "it is not a runtime dependency. No downloads, Turbo adapters or substitutions."
)
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_MEMBER_BYTES = 8 * 1024 * 1024


def qwen_character_sheet_archive_path() -> Path:
    return settings.model_dir / "workflows" / "qwen-character-sheet" / "Character_Sheet.zip"


def qwen_character_sheet_size(megapixels: float = 3.4) -> tuple[int, int]:
    """Exact core ResolutionSelector 3:2, multiple=32 calculation (binary MP)."""
    if (isinstance(megapixels, bool) or not isinstance(megapixels, (float, int))
            or not math.isfinite(megapixels) or not 1 <= megapixels <= 6):
        raise ValueError("Sheet megapixels must be finite and between 1 and 6 (presets 1, 3.4, 6).")
    scale = math.sqrt(megapixels * 1024 * 1024 / 6)
    return round(3 * scale / 32) * 32, round(2 * scale / 32) * 32


def load_qwen_character_sheet_prompts(layout: str = "Simple") -> tuple[str, str]:
    """Read only the two named JSON members; never extract or trust arbitrary paths."""
    if layout not in QWEN_CHARACTER_SHEET_PROFILES:
        raise ValueError("Sheet layout must be Production or Simple.")
    path = qwen_character_sheet_archive_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing pinned character-sheet archive: {path}. {QWEN_CHARACTER_SHEET_ASSET_HINT}")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_ARCHIVE_BYTES + 1)
    if len(raw) > _MAX_ARCHIVE_BYTES:
        raise ValueError("Character-sheet archive exceeds the 32 MiB limit.")
    if hashlib.sha256(raw).hexdigest() != QWEN_CHARACTER_SHEET_SHA256:
        raise ValueError("Character-sheet archive SHA256 mismatch; supply the unaltered pinned archive.")
    documents = {}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        if len(archive.infolist()) > 128:
            raise ValueError("Character-sheet archive has too many members.")
        for filename, _, _ in QWEN_CHARACTER_SHEET_PROFILES.values():
            entries = [item for item in archive.infolist() if item.filename == filename]
            if len(entries) != 1:
                raise ValueError(f"Pinned archive requires exactly one root member {filename}.")
            member = entries[0]
            if (member.is_dir() or member.flag_bits & 1 or member.file_size > _MAX_MEMBER_BYTES
                    or member.file_size / max(member.compress_size, 1) > 200):
                raise ValueError(f"Unsafe or oversized character-sheet ZIP member: {filename}.")
            with archive.open(member) as stream:
                content = stream.read(_MAX_MEMBER_BYTES + 1)
            if len(content) > _MAX_MEMBER_BYTES:
                raise ValueError(f"Character-sheet ZIP member exceeds limit: {filename}.")
            documents[filename] = json.loads(content)
    filename, subgraph_id, static_id = QWEN_CHARACTER_SHEET_PROFILES[layout]
    groups = [group for group in documents[filename]["definitions"]["subgraphs"]
              if group.get("id") == subgraph_id]
    if len(groups) != 1:
        raise ValueError(f"Pinned {layout} subgraph is missing or ambiguous.")

    def value(node_id: int) -> str:
        nodes = [node for node in groups[0]["nodes"] if node.get("id") == node_id]
        if len(nodes) != 1:
            raise ValueError(f"Pinned {layout} prompt node {node_id} is missing or ambiguous.")
        node = nodes[0]
        named = node.get("widgets_values_named", {})
        text = named.get("value") if "value" in named else node.get("widgets_values", [None])[0]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Pinned {layout} prompt node {node_id} has no string value.")
        return text

    return value(478), value(static_id)


def missing_qwen_character_sheet_assets(root: str | Path, prompt_mode: str = "Static") -> list[str]:
    if prompt_mode not in {"Static", "Auto"}:
        raise ValueError("Sheet prompt mode must be Static or Auto.")
    paths = [Path(root) / "models" / name for name in QWEN_CHARACTER_SHEET_FILES]
    paths.append(qwen_character_sheet_archive_path())
    if prompt_mode == "Auto":
        paths.append(Path(root) / "models" / QWEN_CHARACTER_SHEET_PE)
    return [str(path) for path in paths if not path.is_file()]


def validate_qwen_character_sheet_request(req: GenerationRequest) -> None:
    if req.model_key != QWEN_CHARACTER_SHEET_KEY or req.workflow != QWEN_CHARACTER_SHEET_WORKFLOW:
        raise ValueError("Qwen sheets require model_key='qwen-2.1-sheet' and workflow='qwen-character-sheet'.")
    if req.sheet_layout not in QWEN_CHARACTER_SHEET_PROFILES or req.sheet_prompt_mode not in {"Static", "Auto"}:
        raise ValueError("Choose Production/Simple and Static/Auto for Qwen character sheets.")
    if len(req.images) != 1 or req.mask is not None or type(req.count) is not int or req.count != 1:
        raise ValueError("Qwen sheets require exactly one reference, no mask and count=1.")
    if req.size_multiplier != 1 or req.swap_kind is not None:
        raise ValueError("Qwen sheets require multiplier=1 and no swap.")
    if type(req.steps) is not int or req.steps != 25 or req.guidance != 1 or req.true_cfg != 1:
        raise ValueError("Full BF16 Qwen sheets require 25 steps and CFG/true_cfg=1.")
    if req.negative_prompt.strip():
        raise ValueError("Qwen sheets use no negative prompt at CFG 1.")
    if type(req.seed) is not int or not 0 <= req.seed < 2**64:
        raise ValueError("Sheet seed must be an integer in [0, 2**64).")
    for text in (req.prompt, req.sheet_entity_name, req.sheet_character_description):
        if not isinstance(text, str) or "\x00" in text:
            raise ValueError("Sheet prompt/name/description must be strings without NUL characters.")
    qwen_character_sheet_size(req.sheet_megapixels)
    if len(req.loras) > 5:
        raise ValueError("Qwen sheets support at most five optional ModelOnly LoRAs.")
    directory = (settings.lora_dir / "qwen21").resolve()
    seen: set[str] = set()
    for lora in req.loras:
        path = lora.path.resolve()
        if path.parent != directory or path.name != lora.name or path.suffix.lower() != ".safetensors":
            raise ValueError("Sheet LoRAs must be .safetensors files in models/loras/qwen21/.")
        if path.name.casefold() in seen:
            raise ValueError(f"Duplicate sheet LoRA: {path.name}.")
        seen.add(path.name.casefold())
        validate_optional_lora_weight(lora.weight)
        if not path.is_file():
            raise FileNotFoundError(f"Missing optional sheet LoRA: {path}")


def configure_qwen_character_sheet_graph(
    req: GenerationRequest, image_name: str = "source.png",
) -> dict[str, Any]:
    validate_qwen_character_sheet_request(req)
    system, static = load_qwen_character_sheet_prompts(req.sheet_layout)
    width, height = qwen_character_sheet_size(req.sheet_megapixels)
    customization = req.prompt.strip()
    # Never inherit the publisher's example identity from a static sheet.
    static = re.sub(r"\bAyaka\b", lambda _: req.sheet_entity_name.strip() or "the reference entity",
                    static, flags=re.IGNORECASE)
    prompt = static + ("\n\nCustomization:\n" + customization if customization else "")
    graph: dict[str, Any] = {}

    def node(key: str, class_type: str, **inputs: Any) -> None:
        graph[key] = {"class_type": class_type, "inputs": inputs}

    node("1", "UNETLoader", unet_name=QWEN_CHARACTER_SHEET_FILES[0].split("/")[-1], weight_dtype="default")
    previous = "1"
    for index, lora in enumerate(req.loras, start=40):
        if lora.weight == 0:
            continue
        key = str(index)
        node(key, "LoraLoaderModelOnly", model=[previous, 0],
             lora_name=f"qwen21/{lora.name}", strength_model=float(lora.weight))
        previous = key
    node("2", "ModelAttentionBackend", model=[previous, 0], attention="comfy kitchen attention")
    node("3", "QwenImage21Cache", model=["2", 0], device="auto", dtype="default")
    node("4", "CLIPLoader", clip_name=QWEN_CHARACTER_SHEET_FILES[1].split("/")[-1], type="qwen_image", device="default")
    node("5", "VAELoader", vae_name=QWEN_CHARACTER_SHEET_FILES[2].split("/")[-1])
    node("6", "LoadImage", image=image_name)
    # No node 7: the uploaded reference is already aspect-normalized locally.
    node("8", "TextEncodeQwenImage21", clip=["4", 0], vae=["5", 0],
         prompt=prompt, negative_prompt="", resolution=1536, **{"images.image_1": ["6", 0]})
    node("9", "EmptyLatentImage", width=width, height=height, batch_size=1)
    node("10", "KSampler", model=["3", 0], positive=["8", 0], negative=["8", 1],
         latent_image=["9", 0], seed=req.seed, steps=25, cfg=1.0,
         sampler_name="res_multistep", scheduler="beta", denoise=1.0)
    node("11", "VAEDecode", samples=["10", 0], vae=["5", 0])
    node("12", "VAEDeGrid", image=["11", 0], enabled=True, mode="auto", limit=0.02,
         skip_when_clean=True, grid_gain=10, grid_view="4x zoom")
    node("29", "SaveImage", images=["12", 0], filename_prefix=f"photo_edit_qwen_sheet_{uuid.uuid4().hex}")
    if req.sheet_prompt_mode == "Auto":
        auto_prompt = ("Entity name: " + req.sheet_entity_name if req.sheet_layout == "Production"
                       else req.sheet_character_description)
        if customization:
            auto_prompt += "\n\nCustomization:\n" + customization
        node("13", "CLIPLoader", clip_name=QWEN_CHARACTER_SHEET_PE.split("/")[-1], type="stable_diffusion", device="default")
        node("14", "TextGenerate", clip=["13", 0], image=["6", 0], prompt=auto_prompt,
             system_prompt=system, thinking=True, use_default_template=True,
             max_length=2560 if req.sheet_layout == "Production" else 2048, sampling_mode="off")
        graph["8"]["inputs"]["prompt"] = ["14", 0]
    return add_dlss_nodes(graph, req)


def preflight_qwen_character_sheet_graph(graph: dict[str, Any], registered: dict[str, Any]) -> None:
    # Lazy import avoids coupling the new family to Turbo graph configuration.
    from photo_edit_studio.models.comfy_swap import preflight_krea_remix_graph

    missing = {node["class_type"] for node in graph.values()} - registered.keys()
    if missing:
        raise RuntimeError(f"Qwen sheet missing nodes: {', '.join(sorted(missing))}. {QWEN_CHARACTER_SHEET_ASSET_HINT}")
    expanded = deepcopy(registered)
    for entry in expanded.values():
        schema = entry.get("input", {})
        required = schema.setdefault("required", {})
        fields = {**required, **schema.get("optional", {})}
        for group, definition in fields.items():
            if definition[0] != "COMFY_AUTOGROW_V3":
                continue
            template = definition[1].get("template", {})
            template_input = template.get("input", {})
            definitions = {**template_input.get("required", {}), **template_input.get("optional", {})}
            if len(definitions) == 1:
                for name in template.get("names", []):
                    required[f"{group}.{name}"] = next(iter(definitions.values()))
    preflight_krea_remix_graph(graph, expanded, label="Qwen character sheet")
    sampler = expanded["KSampler"]["input"]["required"]["sampler_name"]
    choices = sampler[1].get("options", []) if sampler[0] == "COMBO" else sampler[0]
    if not isinstance(choices, list) or "res_multistep" not in choices:
        raise RuntimeError("Qwen sheets require registered res_multistep; no sampler substitution.")


class ComfyQwenCharacterSheetAdapter(ModelAdapter):
    def load(self) -> None:
        raise RuntimeError("Qwen character sheets run through the local ComfyUI API.")

    def generate(self, request: GenerationRequest, progress: GenerationProgress | None = None) -> list[Image.Image]:
        from photo_edit_studio.comfy_backend import _endpoint, ensure_backend

        validate_qwen_character_sheet_request(request)
        validate_dlss_request(request)
        if self.spec.key != request.model_key:
            raise ValueError("Sheet adapter model key does not match request.")
        missing = missing_qwen_character_sheet_assets(settings.comfy_dir, request.sheet_prompt_mode)
        if missing:
            raise FileNotFoundError("Qwen sheet assets missing: " + ", ".join(missing) + ". " + QWEN_CHARACTER_SHEET_ASSET_HINT)
        graph = configure_qwen_character_sheet_graph(request)
        if not cuda_available() or cuda_vram_gb() + 0.5 < self.spec.minimum_vram_gb:
            raise RuntimeError("Full BF16 Qwen sheets require CUDA, about 12 GB VRAM, CPU offload and ample host RAM.")
        _endpoint(settings.comfy_url)
        ensure_backend(model_key=request.model_key, workflow=request.workflow,
                       sheet_prompt_mode=request.sheet_prompt_mode)
        with httpx.Client(base_url=settings.comfy_url.rstrip("/"), timeout=60, trust_env=False) as client:
            client.get("/system_stats").raise_for_status()
            response = client.get("/object_info")
            response.raise_for_status()
            registered = response.json()
            if not isinstance(registered, dict):
                raise RuntimeError("ComfyUI object_info must be a node registry dictionary.")  # noqa: TRY004 -- malformed API response
            preflight_qwen_character_sheet_graph(graph, registered)
            data = BytesIO()
            normalize_image(request.images[0], max_side=1536).save(data, format="PNG")
            response = client.post("/upload/image", files={"image": (
                f"qwen_sheet_{uuid.uuid4().hex}.png", data.getvalue(), "image/png")},
                data={"type": "input", "overwrite": "false"})
            response.raise_for_status()
            upload = response.json()
            graph["6"]["inputs"]["image"] = "/".join(
                part for part in (upload.get("subfolder", ""), upload["name"]) if part)
            response = client.post("/prompt", json={"prompt": graph})
            if response.is_error:
                raise RuntimeError(f"ComfyUI rejected Qwen sheet: {response.text[:500]}")
            prompt_id = response.json()["prompt_id"]
            deadline = time.monotonic() + settings.comfy_timeout_seconds
            while time.monotonic() < deadline:
                response = client.get(f"/history/{prompt_id}")
                response.raise_for_status()
                job = response.json().get(prompt_id, {})
                if job.get("status", {}).get("status_str") == "error":
                    raise RuntimeError("Qwen sheet failed in ComfyUI; see logs/comfyui.log.")
                outputs = job.get("outputs", {}).get("29", {}).get("images", [])
                if outputs:
                    if len(outputs) != 1:
                        raise RuntimeError("Qwen sheets must return exactly one SaveImage 29 output.")
                    response = client.get("/view", params={
                        "filename": outputs[0]["filename"], "subfolder": outputs[0].get("subfolder", ""),
                        "type": outputs[0].get("type", "output")})
                    response.raise_for_status()
                    with Image.open(BytesIO(response.content)) as image:
                        result = image.convert("RGB")
                    if not validate_dlss_request(request) and result.size != qwen_character_sheet_size(request.sheet_megapixels):
                        raise RuntimeError("Qwen sheet output differs from native canvas; refusing resizing.")
                    return [result]
                if progress is not None:
                    progress.update(0.5, "Generating full BF16 Qwen character sheet")
                time.sleep(0.5)
        raise TimeoutError("Qwen character-sheet generation timed out.")