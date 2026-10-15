"""An explicitly-populated registry of reference configurations.

THE REGISTRY IS EMPTY AT IMPORT AND STAYS EMPTY UNTIL A CALLER FILLS IT.

There is no default configuration, no search path, no environment variable and
no bundled fallback. Asking for an id that nobody registered raises, and the
message says so rather than quietly producing the only configuration that
happens to exist. That is the whole point: production behaviour must be
identical whether or not this package is installed, until something explicitly
asks for a configuration by name.

Configuration data lives with the campaign that froze it, not inside the library.
The library ships the loader, the schema and the validators; the campaign supplies
the document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import ReferenceConfigurationError
from .loader import load_reference_configuration
from .model import ReferenceConfiguration

__all__ = [
    "DEFAULT_REFERENCE_CONFIGURATION",
    "ReferenceConfigurationRegistry",
    "RegistryEntry",
    "clear_registry",
    "get_reference_configuration",
    "register_configuration_directory",
    "register_configuration_file",
    "registered_ids",
]

#: There is no default. Stated as a name so the absence is greppable, and so a
#: future edit that introduces one is visible in a diff.
DEFAULT_REFERENCE_CONFIGURATION: None = None


@dataclass(frozen=True)
class RegistryEntry:
    """What is known about a configuration without loading it.

    ``configuration_class`` is exposed here so a caller can see that an entry is
    a public reference rather than a mission configuration before deciding to
    use it. Being in the registry implies nothing about being flight truth.
    """

    configuration_id: str
    configuration_class: str
    path: Path
    expected_ancestor_fingerprint: str | None = None


class ReferenceConfigurationRegistry:
    """A mapping of configuration id to a file, populated only on request."""

    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._loaded: dict[str, ReferenceConfiguration] = {}

    # -- population -----------------------------------------------------
    def register_file(
        self,
        path: str | Path,
        *,
        expected_ancestor_fingerprint: str | None = None,
    ) -> RegistryEntry:
        """Register one configuration document.

        The id and class are read from the document rather than supplied by the
        caller, so a registry entry cannot disagree with the file it points at.
        """
        p = Path(path)
        try:
            document = json.loads(p.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ReferenceConfigurationError(
                f"cannot register {p}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ReferenceConfigurationError(
                f"cannot register {p}: not valid JSON ({exc})"
            ) from exc

        try:
            identity = document["identity"]
            cid = str(identity["configuration_id"]["value"])
            cclass = str(identity["configuration_class"]["value"])
        except (KeyError, TypeError) as exc:
            raise ReferenceConfigurationError(
                f"cannot register {p}: no readable identity.configuration_id / "
                f"configuration_class ({exc})"
            ) from exc

        existing = self._entries.get(cid)
        if existing is not None and existing.path != p:
            raise ReferenceConfigurationError(
                f"configuration id {cid!r} is already registered from "
                f"{existing.path}; refusing to shadow it with {p}"
            )

        entry = RegistryEntry(
            configuration_id=cid,
            configuration_class=cclass,
            path=p,
            expected_ancestor_fingerprint=expected_ancestor_fingerprint,
        )
        self._entries[cid] = entry
        self._loaded.pop(cid, None)
        return entry

    def register_directory(self, path: str | Path, pattern: str = "*.json") -> tuple[RegistryEntry, ...]:
        """Register every readable configuration document in a directory.

        Files that are not configuration documents are skipped silently — a
        data directory holds many kinds of JSON — but a file that looks like a
        configuration and fails to register raises.
        """
        d = Path(path)
        if not d.is_dir():
            raise ReferenceConfigurationError(f"{d} is not a directory")
        out: list[RegistryEntry] = []
        for candidate in sorted(d.glob(pattern)):
            try:
                document = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(document, dict) or "identity" not in document:
                continue
            out.append(self.register_file(candidate))
        return tuple(out)

    # -- access ---------------------------------------------------------
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def describe(self, configuration_id: str) -> RegistryEntry:
        try:
            return self._entries[configuration_id]
        except KeyError:
            raise self._unregistered(configuration_id) from None

    def get(self, configuration_id: str) -> ReferenceConfiguration:
        """Load (once) and return a registered configuration."""
        if configuration_id in self._loaded:
            return self._loaded[configuration_id]
        entry = self.describe(configuration_id)
        config = load_reference_configuration(
            entry.path,
            expected_ancestor_fingerprint=entry.expected_ancestor_fingerprint,
        )
        if config.configuration_id != configuration_id:
            raise ReferenceConfigurationError(
                f"{entry.path} declares id {config.configuration_id!r} but was "
                f"registered as {configuration_id!r}"
            )
        self._loaded[configuration_id] = config
        return config

    def clear(self) -> None:
        self._entries.clear()
        self._loaded.clear()

    def _unregistered(self, configuration_id: str) -> ReferenceConfigurationError:
        known = ", ".join(self.ids()) or "nothing"
        return ReferenceConfigurationError(
            f"no configuration registered as {configuration_id!r}; currently "
            f"registered: {known}. Reference configurations are opt-in: register "
            "the document explicitly with register_configuration_file(path). "
            "There is no default configuration and no search path."
        )


#: The process-wide registry. Empty until a caller registers something.
_REGISTRY = ReferenceConfigurationRegistry()


def register_configuration_file(
    path: str | Path, *, expected_ancestor_fingerprint: str | None = None
) -> RegistryEntry:
    return _REGISTRY.register_file(
        path, expected_ancestor_fingerprint=expected_ancestor_fingerprint
    )


def register_configuration_directory(
    path: str | Path, pattern: str = "*.json"
) -> tuple[RegistryEntry, ...]:
    return _REGISTRY.register_directory(path, pattern)


def get_reference_configuration(configuration_id: str) -> ReferenceConfiguration:
    return _REGISTRY.get(configuration_id)


def registered_ids() -> tuple[str, ...]:
    return _REGISTRY.ids()


def clear_registry() -> None:
    _REGISTRY.clear()
