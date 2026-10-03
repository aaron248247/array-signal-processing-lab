"""Automated consistency checks for the documentation.

Checks:
  1. every relative Markdown link and image in README.md and docs/*.md points to an existing file,
     and every #anchor exists in the target file (GitHub slug rules);
  2. every figure file in figures/ is embedded in docs/07_figure_index.md and docs/04_results.md,
     and every embedded figure exists;
  3. figure numbering is continuous (1..n) and in order in 04_results.md and 07_figure_index.md,
     and "Figure k" is always paired with figNN_ where NN = k;
  4. the figure filenames written by the plotting code are exactly the files in figures/;
  5. every numbered citation [n] in docs/*.md and README.md exists in docs/06_references.md,
     and every reference is cited at least once;
  6. no stale names: no reference to a figures/*.png, docs/*.md or experiments/*.py that does not exist.

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md"))
MD_FILES = [ROOT / "README.md"] + DOCS
errors: list[str] = []


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)  # drop punctuation (keeps letters, digits, _, -, space)
    return s.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    out, seen = set(), {}
    in_code = False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            base = slug(m.group(2))
            n = seen.get(base, 0)
            out.add(base if n == 0 else f"{base}-{n}")
            seen[base] = n + 1
    return out


LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)|<img[^>]*src=\"([^\"]+)\"")

# 1. links
for md in MD_FILES:
    text = md.read_text()
    for m in LINK.finditer(text):
        target = m.group(1) or m.group(2)
        if re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
            continue
        path, _, frag = target.partition("#")
        dest = (md.parent / path).resolve() if path else md
        if not dest.exists():
            errors.append(f"{md.relative_to(ROOT)}: broken link -> {target}")
            continue
        if frag and dest.suffix == ".md" and frag not in anchors(dest):
            errors.append(f"{md.relative_to(ROOT)}: missing anchor #{frag} in {dest.relative_to(ROOT)}")

# 2./3. figures
fig_files = sorted(p.name for p in (ROOT / "figures").glob("*.png"))
for doc in ("04_results.md", "07_figure_index.md"):
    text = (ROOT / "docs" / doc).read_text()
    embedded = re.findall(r"!\[Figure (\d+)\]\(\.\./figures/(fig(\d\d)_[^)]+\.png)\)", text)
    nums = [int(a) for a, _, _ in embedded]
    if nums != list(range(1, len(fig_files) + 1)):
        errors.append(f"docs/{doc}: embedded figure numbers {nums} are not 1..{len(fig_files)} in order")
    for a, fname, nn in embedded:
        if int(a) != int(nn):
            errors.append(f"docs/{doc}: 'Figure {a}' embeds {fname}")
        if fname not in fig_files:
            errors.append(f"docs/{doc}: embeds missing file {fname}")
    missing = set(fig_files) - {f for _, f, _ in embedded}
    for f in sorted(missing):
        errors.append(f"docs/{doc}: figure file {f} not embedded")
    # bold captions "**Figure k.**" in 04 must follow the same order
    if doc == "04_results.md":
        caps = [int(x) for x in re.findall(r"\*\*Figure (\d+)\.\*\*", text)]
        if caps != list(range(1, len(fig_files) + 1)):
            errors.append(f"docs/{doc}: caption numbers {caps} not continuous")
    if doc == "07_figure_index.md":
        heads = [int(x) for x in re.findall(r"^## Figure (\d+) ", text, flags=re.M)]
        if heads != list(range(1, len(fig_files) + 1)):
            errors.append(f"docs/{doc}: section numbers {heads} not continuous")

# README figure numbers must match file numbers
readme = (ROOT / "README.md").read_text()
for fname, num in re.findall(r'src="figures/(fig\d\d_[^"]+)"[^>]*alt="Figure (\d+)"', readme):
    if int(num) != int(fname[3:5]):
        errors.append(f"README.md: alt 'Figure {num}' on {fname}")
readme_caps = [int(x) for x in re.findall(r"\*\*Figure (\d+)\.\*\*", readme)]
readme_imgs = [int(f[3:5]) for f in re.findall(r'src="figures/(fig\d\d_[^"]+)"', readme)]
if readme_caps != readme_imgs:
    errors.append(f"README.md: captions {readme_caps} do not match images {readme_imgs}")

# 4. figure names written by code
code_names = set()
for py in (ROOT / "experiments").glob("*.py"):
    code_names |= set(re.findall(r"save\(fig,\s*\"(fig\d\d_[^\"]+\.png)\"\)", py.read_text()))
if code_names != set(fig_files):
    errors.append(f"plotting code writes {sorted(code_names ^ set(fig_files))} which differ from figures/")

# 5. citations
ref_text = (ROOT / "docs" / "06_references.md").read_text()
ref_nums = {int(x) for x in re.findall(r"^(\d+)\.\s", ref_text, flags=re.M)}
cited = set()
CITE = re.compile(r"(?<![\w!\]])\[(\d+(?:\s*,\s*\d+)*)\](?!\()")
for md in MD_FILES:
    text = md.read_text()
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"\$[^$\n]*\$", "", text)
    for m in CITE.finditer(text):
        for n in re.split(r"\s*,\s*", m.group(1)):
            n = int(n)
            if md.name != "06_references.md":
                cited.add(n)
            if n not in ref_nums:
                errors.append(f"{md.relative_to(ROOT)}: citation [{n}] not in references")
for n in sorted(ref_nums - cited):
    errors.append(f"docs/06_references.md: reference [{n}] is never cited in README/docs")
if sorted(ref_nums) != list(range(1, len(ref_nums) + 1)):
    errors.append("docs/06_references.md: reference numbering not continuous")

# 6. stale names anywhere in text files
TEXT = list(ROOT.glob("*.md")) + DOCS + list((ROOT / "experiments").glob("*.py")) + list((ROOT / "src").rglob("*.py"))
for f in TEXT:
    t = f.read_text()
    for name in set(re.findall(r"fig\d\d_[A-Za-z0-9_]+\.png", t)):
        if name not in fig_files:
            errors.append(f"{f.relative_to(ROOT)}: stale figure name {name}")
    for name in set(re.findall(r"\b(0\d_[a-z_]+\.md)\b", t)):
        if not (ROOT / "docs" / name).exists():
            errors.append(f"{f.relative_to(ROOT)}: stale doc name {name}")
    for name in set(re.findall(r"experiments/([a-z_]+\.py)", t)):
        if not (ROOT / "experiments" / name).exists():
            errors.append(f"{f.relative_to(ROOT)}: stale script name experiments/{name}")

n_links = sum(len(LINK.findall(md.read_text())) for md in MD_FILES)
print(f"checked {len(MD_FILES)} markdown files, {n_links} links/images, {len(fig_files)} figures, "
      f"{len(ref_nums)} references ({len(cited)} cited)")
if errors:
    print(f"{len(errors)} problem(s):")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("all documentation checks passed")
