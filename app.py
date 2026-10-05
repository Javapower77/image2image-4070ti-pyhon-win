from __future__ import annotations

import logging
from pathlib import Path

from photo_edit_studio.diagnostics import configure_logging


def main() -> None:
    configure_logging()
    logger = logging.getLogger("photo_edit_studio.app")
    logger.info("Application startup; importing configuration and UI")
    from photo_edit_studio.config import settings
    from photo_edit_studio.ui import CSS, build_app, build_theme

    auth = None
    if settings.auth_user and settings.auth_password:
        auth = (settings.auth_user, settings.auth_password)
    logger.info("Building Gradio UI")
    app = build_app()
    logger.info("Launching Gradio server on port %s", settings.port)
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
    logger.warning("Gradio launch returned; is_running=%s", app.is_running)


if __name__ == "__main__":
    main()
