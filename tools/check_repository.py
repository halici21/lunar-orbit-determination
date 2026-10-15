"""Check repository paths, preserved scientific bytes and current documentation links."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / 'docs/development/path-migration.json').read_text(encoding='utf-8'))
    errors = []
    preserved = 0
    for old, item in manifest['files'].items():
        path = ROOT / item['path']
        if not path.is_file():
            errors.append(f'Missing migrated file: {item["path"]} (was {old})')
        elif item.get('byte_preserved'):
            preserved += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item['original_sha256']:
                errors.append(f'Preserved scientific bytes changed: {path.relative_to(ROOT)}')

    # Archived snapshots and synchronized historical skill copies are immutable
    # records, not current navigation. Their paths resolve through the manifest.
    documents = [ROOT / name for name in ('README.md', 'CONTRIBUTING.md', 'CHANGELOG.md')]
    for folder in ('docs', 'studies', 'data', 'tests', 'examples', 'desktop_app'):
        documents.extend((ROOT / folder).rglob('*.md'))
    link_count = 0
    pattern = re.compile(r'(?<![\w])!?\[[^\]\n]*\]\(([^\s)]+)\)')
    for document in documents:
        source = document.read_text(encoding='utf-8')
        # Examples of indexing and math are not Markdown navigation.
        prose = re.sub(r'```[\s\S]*?```|`[^`\n]+`', lambda m: re.sub(r'[^\n]', ' ', m[0]), source)
        for match in pattern.finditer(prose):
            target = match[1]
            if target.startswith(('#', '/')) or re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', target):
                continue
            target = unquote(target.split('#', 1)[0])
            link_count += 1
            if not (document.parent / target).exists():
                line = source[:match.start()].count('\n') + 1
                errors.append(f'{document.relative_to(ROOT)}:{line}: broken link {target}')

    import lunar_od
    if not Path(lunar_od.__file__).resolve().is_relative_to(ROOT / 'src/lunar_od'):
        errors.append(f'Library imported from another checkout: {lunar_od.__file__}')
    if (ROOT / 'lunar_od/__init__.py').is_file():
        errors.append('Obsolete root package shadows the src package')
    gallery = ROOT / 'docs/figures/quickstart'
    if gallery.is_dir():
        summary = json.loads((gallery / 'summary.json').read_text(encoding='utf-8'))
        for name, digest in summary['artifact_sha256'].items():
            path = gallery / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                errors.append(f'Curated tutorial artifact changed: {name}')
        source = json.loads((gallery / 'visibility-source.json').read_text(encoding='utf-8'))
        if hashlib.sha256((ROOT / source['source']).read_bytes()).hexdigest() != source['source_sha256']:
            errors.append('Curated visibility fixture identity changed')
    print(f'{len(manifest["files"])} migrated paths; {preserved} preserved byte identities; '
          f'{len(documents)} documents / {link_count} local links checked.')
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
