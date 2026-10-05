from photo_edit_studio.dlss import DLSS_DEFAULTS
from photo_edit_studio.ui import _dlss_controls_changed, _with_dlss, build_app


def test_dlss_disabled_and_enabled_controls():
    values = {**DLSS_DEFAULTS, "enabled": False}
    assert _dlss_controls_changed(*values.values()) is None
    values["enabled"] = True
    assert _dlss_controls_changed(*values.values()) == DLSS_DEFAULTS


def test_dlss_wrapper_preserves_callback_arguments():
    calls = []

    def callback(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    assert _with_dlss(callback)("source", "prompt", {"nr_style": "Natural"}, progress="notify") == "result"
    assert calls == [(("source", "prompt"), {"dlss": {"nr_style": "Natural"}, "progress": "notify"})]


def test_dlss_checkbox_disabled_by_default():
    components = build_app().config["components"]
    checkbox = next(component for component in components
                    if component["type"] == "checkbox" and component["props"].get("label") == "Enabled")
    assert checkbox["props"]["value"] is False