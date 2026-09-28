from __future__ import annotations

from pathlib import Path

from photo_edit_studio.config import settings
from photo_edit_studio.ui import CSS, build_app, build_theme


def main() -> None:
    auth = None
    if settings.auth_user and settings.auth_password:
        auth = (settings.auth_user, settings.auth_password)
    app = build_app()
    app.queue(default_concurrency_limit=settings.max_concurrency, max_size=8).launch(
        server_name=settings.host,
        server_port=settings.port,
        share=settings.share,
        auth=auth,
        show_error=True,
        css=CSS,
        theme=build_theme(),
        head="<script>" + (Path(__file__).resolve().parent / "photo_edit_studio" / "assets" / "ui" / "studio-popovers.js").read_text(encoding="utf-8") + "</script>",
    )


if __name__ == "__main__":
    main()
