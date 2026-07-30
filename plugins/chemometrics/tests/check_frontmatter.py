#!/usr/bin/env python
"""Validate each SKILL.md frontmatter: name + description present, kebab-case,
unique names, and description length within the recommended budget.

Run: ``python plugins/chemometrics/tests/check_frontmatter.py``
"""

from __future__ import annotations

import re
from pathlib import Path

SKILLS = Path(__file__).resolve().parents[1] / "skills"
KEBAB = re.compile(r"^[a-z][a-z0-9-]*$")
MAX_DESCRIPTION = 1024


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        raise ValueError("missing YAML frontmatter")
    end = text.index("\n---", 3)
    block = text[3:end]
    # minimal YAML: top-level `key:` with optional folded `>-` scalar
    fields: dict[str, str] = {}
    key = None
    buf: list[str] = []
    for line in block.splitlines():
        m = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if m and not line.startswith(" "):
            if key:
                fields[key] = " ".join(buf).strip()
            key, rest = m.group(1), m.group(2).strip()
            buf = [] if rest in (">-", "|", ">", "") else [rest]
        elif key:
            buf.append(line.strip())
    if key:
        fields[key] = " ".join(buf).strip()
    return fields


# The one cross-domain orchestrator; every other skill is a narrow spoke. The
# orchestrator's description is broad on purpose, so it is exempt from the
# spoke-vs-spoke overlap check.
ORCHESTRATOR = "chemometrics-workflow"
# Words too generic to signal a shared trigger.
_STOP = {
    "the", "and", "for", "with", "use", "when", "one", "each", "into", "via", "not",
    "you", "your", "are", "them", "this", "that", "from", "per", "a", "an", "of", "to",
    "or", "on", "in", "it", "is", "as", "by", "at", "run", "used", "using", "which",
    "chemotools", "scikit", "learn", "sklearn", "spectra", "spectral", "chemometric",
    "chemometrics", "model", "models", "data", "skill", "delegates", "delegate",
}
OVERLAP_LIMIT = 0.5  # Jaccard on keyword sets; spokes must stay below this


def _keywords(desc: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", desc.lower()) if w not in _STOP}


def main() -> None:
    names: dict[str, Path] = {}
    descs: dict[str, str] = {}
    problems: list[str] = []
    skill_files = sorted(SKILLS.glob("*/SKILL.md"))
    if not skill_files:
        raise SystemExit("no SKILL.md files found")

    for path in skill_files:
        fm = _parse_frontmatter(path.read_text(encoding="utf-8"))
        name = fm.get("name", "")
        desc = fm.get("description", "")
        if not name:
            problems.append(f"{path}: missing name")
        elif not KEBAB.match(name):
            problems.append(f"{path}: name {name!r} is not kebab-case")
        elif name in names:
            problems.append(f"{path}: duplicate name {name!r} (also {names[name]})")
        else:
            names[name] = path
            descs[name] = desc
        if not desc:
            problems.append(f"{path}: missing description")
        elif len(desc) > MAX_DESCRIPTION:
            problems.append(f"{path}: description {len(desc)} chars > {MAX_DESCRIPTION}")
        # folder name should match the skill name
        if name and path.parent.name != name:
            problems.append(f"{path}: folder {path.parent.name!r} != name {name!r}")
        print(f"{name:<28} desc={len(desc):>4} chars  ({path.parent.name})")

    # Trigger hygiene: no two spokes should claim near-identical trigger space.
    spokes = sorted(n for n in descs if n != ORCHESTRATOR)
    worst = 0.0
    for i, a in enumerate(spokes):
        ka = _keywords(descs[a])
        for b in spokes[i + 1:]:
            kb = _keywords(descs[b])
            union = ka | kb
            jac = len(ka & kb) / len(union) if union else 0.0
            worst = max(worst, jac)
            if jac > OVERLAP_LIMIT:
                problems.append(
                    f"trigger overlap: {a!r} vs {b!r} Jaccard={jac:.2f} "
                    f"> {OVERLAP_LIMIT} (shared: {sorted(ka & kb)})"
                )

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    print(f"\nOK: {len(names)} skills, names unique and well-formed")
    print(f"    trigger overlap max (spoke vs spoke) = {worst:.2f} (limit {OVERLAP_LIMIT})")


if __name__ == "__main__":
    main()
