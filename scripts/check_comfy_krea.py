from __future__ import annotations

import argparse
import json

import httpx

from photo_edit_studio.comfy_assets import missing_assets
from photo_edit_studio.comfy_workflows import krea_reference_template
from photo_edit_studio.config import settings
from photo_edit_studio.models.comfy_swap import _api_graph, _local_comfy_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local ComfyUI Krea reference readiness.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable check results")
    args = parser.parse_args()
    checks: dict[str, str] = {}
    path = settings.comfy_reference_workflow
    if path.is_file():
        try:
            graph = _api_graph(json.loads(path.read_text(encoding="utf-8")))
            required = ("72", "84", "85", "79", "82", "53", "29", "71")
            missing = [node for node in required if node not in graph]
            checks["workflow"] = "ready" if not missing else f"missing nodes: {missing}"
            checks["two_references"] = (
                "ready" if "90" in graph and "image_b" in graph.get("84", {}).get("inputs", {})
                else "second image group not enabled"
            )
        except (ValueError, json.JSONDecodeError) as exc:
            checks["workflow"] = str(exc)
    else:
        graph = krea_reference_template()
        checks["workflow"] = "ready (built-in Krea2Edit template)"
        checks["two_references"] = "ready" if "90" in graph else "missing second image"
    base = settings.comfy_dir
    checks["installation"] = (
        "ready" if (base / "main.py").is_file()
        and (base / "custom_nodes" / "comfyui-krea2edit" / "__init__.py").is_file()
        else f"missing embedded ComfyUI or Krea2Edit nodes under {base}"
    )
    missing_models = missing_assets(base)
    checks["models"] = "ready" if not missing_models else f"missing: {missing_models}"
    try:
        url = _local_comfy_url(settings.comfy_url)
        with httpx.Client(base_url=url, timeout=5.0, trust_env=False) as client:
            response = client.get("/system_stats")
            response.raise_for_status()
        checks["comfyui"] = "ready"
    except (ValueError, httpx.HTTPError) as exc:
        checks["comfyui"] = str(exc)
    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        for key, value in checks.items():
            print(f"{key}: {value}")
    server_required = not settings.comfy_autostart
    if (
        not checks.get("workflow", "").startswith("ready")
        or checks.get("installation") != "ready"
        or checks.get("models") != "ready"
        or (server_required and checks.get("comfyui") != "ready")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()