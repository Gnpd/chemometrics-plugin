#!/usr/bin/env python
"""Lint the plugin + marketplace manifests and the plugin's component layout.

Checks that both manifests parse, carry their required fields, that the
marketplace ``source`` points at the real plugin payload, and that every skill
folder is well-formed (has a ``SKILL.md``). Portability check: no skill ``scripts``
file hardcodes an absolute or plugin-rooted path.

Run: ``python plugins/chemometrics/tests/check_manifests.py``
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]          # plugins/chemometrics
REPO_ROOT = PLUGIN_ROOT.parents[1]                          # marketplace root
PLUGIN_MANIFEST = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
MARKET_MANIFEST = REPO_ROOT / ".claude-plugin" / "marketplace.json"
KEBAB = re.compile(r"^[a-z][a-z0-9-]*$")


def _frontmatter(text: str) -> set[str]:
    """Return the set of top-level frontmatter keys in a markdown component."""
    if not text.startswith("---"):
        return set()
    try:
        end = text.index("\n---", 3)
    except ValueError:
        return set()
    keys = set()
    for line in text[3:end].splitlines():
        m = re.match(r"^(\w[\w-]*):", line)
        if m and not line.startswith((" ", "\t")):
            keys.add(m.group(1))
    return keys


def _load_json(path: Path):
    if not path.exists():
        raise SystemExit(f"missing manifest: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")


def main() -> None:
    problems: list[str] = []

    plugin = _load_json(PLUGIN_MANIFEST)
    for field in ("name", "description"):
        if not plugin.get(field):
            problems.append(f"plugin.json: missing required field {field!r}")
    if plugin.get("name") and not KEBAB.match(plugin["name"]):
        problems.append(f"plugin.json: name {plugin['name']!r} is not kebab-case")
    if "version" not in plugin:
        problems.append("plugin.json: no version (updates require a bumped version)")

    market = _load_json(MARKET_MANIFEST)
    if not market.get("name"):
        problems.append("marketplace.json: missing name")
    entries = market.get("plugins", [])
    if not entries:
        problems.append("marketplace.json: no plugins listed")
    for entry in entries:
        if not entry.get("name") or not entry.get("source"):
            problems.append(f"marketplace.json: entry needs name + source: {entry}")
            continue
        source = (REPO_ROOT / entry["source"]).resolve()
        if not (source / ".claude-plugin" / "plugin.json").exists():
            problems.append(
                f"marketplace.json: source {entry['source']!r} has no plugin.json"
            )

    # Component layout: skills/ must exist with well-formed skill folders.
    skills_dir = PLUGIN_ROOT / "skills"
    if not skills_dir.is_dir():
        problems.append("plugin: skills/ directory missing")
    else:
        skill_dirs = [p for p in skills_dir.iterdir() if p.is_dir()]
        for sd in skill_dirs:
            if not (sd / "SKILL.md").exists():
                problems.append(f"skill {sd.name!r}: no SKILL.md")
        print(f"skills: {len(skill_dirs)} folders, all with SKILL.md"
              if all((sd / 'SKILL.md').exists() for sd in skill_dirs)
              else "skills: some folders missing SKILL.md")

    # Optional component dirs — fine to be empty, but if present must be a dir
    # and any markdown component must carry the frontmatter its type needs.
    for comp in ("commands", "agents"):
        p = PLUGIN_ROOT / comp
        if p.exists() and not p.is_dir():
            problems.append(f"plugin: {comp} exists but is not a directory")
    for cmd in (PLUGIN_ROOT / "commands").glob("*.md"):
        fm = _frontmatter(cmd.read_text(encoding="utf-8"))
        if "description" not in fm:
            problems.append(f"command {cmd.name!r}: frontmatter needs a description")
    for agent in (PLUGIN_ROOT / "agents").glob("*.md"):
        fm = _frontmatter(agent.read_text(encoding="utf-8"))
        for field in ("name", "description"):
            if field not in fm:
                problems.append(f"agent {agent.name!r}: frontmatter needs {field}")

    # Portability: no absolute / plugin-rooted paths baked into scripts.
    bad_paths = re.compile(r"(/plugins/chemometrics/|[A-Za-z]:\\\\Users\\\\|\.claude-plugin)")
    for script in skills_dir.rglob("scripts/*.py"):
        text = script.read_text(encoding="utf-8")
        if bad_paths.search(text):
            problems.append(f"portability: {script.relative_to(PLUGIN_ROOT)} hardcodes a path")

    print(f"plugin: {plugin.get('name')} v{plugin.get('version')}")
    print(f"marketplace: {market.get('name')} ({len(entries)} plugin(s))")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print("\nOK: manifests parse, source resolves, layout + portability clean")


if __name__ == "__main__":
    main()
