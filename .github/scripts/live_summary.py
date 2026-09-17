"""Summarise a pytest JUnit XML report as Markdown, grouped by provider.

Usage: python .github/scripts/live_summary.py live-results.xml > live-summary.md
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def provider(classname: str) -> str:
    """'tests.sources.test_noaa_coops' -> 'noaa_coops'; other modules keep their test name."""
    module = classname.split(".")[2] if classname.startswith("tests.sources.") else classname
    module = module.split(".")[-1] if "." in module else module
    return module.removeprefix("test_")


def summarise(path: Path) -> str:
    if not path.exists():
        return "## Live tests\n\nNo results: the job failed before the tests ran.\n"
    root = ET.parse(path).getroot()
    counts: Counter[str] = Counter()
    failures: list[tuple[str, str, str]] = []
    for case in root.iter("testcase"):
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        skipped = case.find("skipped")
        if problem is not None:
            counts["failed"] += 1
            message = (problem.get("message") or problem.text or "").strip().splitlines()
            failures.append(
                (provider(case.get("classname", "")), case.get("name", ""), (message or [""])[0])
            )
        elif skipped is not None:
            kind = "xfailed" if skipped.get("type") == "pytest.xfail" else "skipped"
            counts[kind] += 1
        else:
            counts["passed"] += 1

    lines = ["## Live tests", ""]
    lines.append(
        ", ".join(f"{counts[k]} {k}" for k in ("passed", "failed", "xfailed", "skipped")) + "."
    )
    if failures:
        providers = sorted({p for p, _, _ in failures})
        lines += ["", f"**Failing providers:** {', '.join(providers)}", ""]
        lines += ["| Provider | Test | Message |", "|---|---|---|"]
        for prov, name, message in failures:
            message = message.replace("|", "\\|")[:200]
            lines.append(f"| {prov} | `{name}` | {message} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(summarise(Path(sys.argv[1] if len(sys.argv) > 1 else "live-results.xml")), end="")
