"""Build a curated documentation site from canonical repository Markdown.

python tools/build_docs.py --strict
Requires requirements/docs.txt; no scientific dependencies or external data.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://github.com/halici21/lunar-orbit-determination/blob/main/"
PAGES = {
    "index.md": "docs/site/index.md",
    "quickstart.md": "docs/tutorials/quickstart.md",
    "results.md": "docs/research/results-gallery.md",
    "benchmark.md": "docs/research/paired-benchmark.md",
    "support.md": "docs/validation/support-matrix.md",
    "theory.md": "docs/site/theory.md",
    "validation.md": "docs/site/validation.md",
    "api.md": "docs/site/api.md",
    "studies.md": "docs/site/studies.md",
    "roadmap.md": "ROADMAP.md",
    "contributing.md": "CONTRIBUTING.md",
}


def build(strict):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from mkdocs.config import load_config
    from mkdocs.commands.build import build as mkdocs_build
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    reverse = {(ROOT / source).resolve(): target for target, source in PAGES.items()}
    with tempfile.TemporaryDirectory(prefix="docs-source-", dir=output) as temporary:
        staging = Path(temporary)
        for target, source in PAGES.items():
            document = ROOT / source

            def rewrite(match):
                label, link = match.groups()
                if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|#", link):
                    return match[0]
                path_part, separator, anchor = link.partition("#")
                resolved = (document.parent / unquote(path_part)).resolve()
                if not resolved.is_relative_to(ROOT) or not resolved.is_file():
                    raise ValueError(f"Missing or outside-repository link in {source}: {link}")
                if resolved in reverse:
                    destination = reverse[resolved]
                elif label.startswith("!"):
                    destination = "assets/" + resolved.relative_to(ROOT).as_posix()
                    asset = staging / destination
                    asset.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(resolved, asset)
                else:
                    destination = SOURCE + resolved.relative_to(ROOT).as_posix()
                return f"{label}({destination}{separator}{anchor})"

            content = re.sub(r"(!?\[[^\]]*\])\(([^\s)]+)\)", rewrite,
                             document.read_text(encoding="utf-8"))
            if target == "api.md":
                from api_reference import render_reference
                content += "\n" + render_reference(ROOT)
            (staging / target).write_text(content, encoding="utf-8")
        assets = staging / "assets"
        assets.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "docs/site/site.css", assets / "site.css")
        config = load_config(config_file=str(ROOT / "mkdocs.yml"),
                             docs_dir=str(staging), strict=strict)
        mkdocs_build(config)
    print(f"Built {len(PAGES)} curated pages at {output / 'site'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    build(parser.parse_args().strict)
