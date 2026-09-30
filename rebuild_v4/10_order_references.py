"""Order the handwritten elsarticle bibliography by first in-text citation."""
from pathlib import Path
import re

tex = Path(__file__).resolve().parents[2] / "AE_paper_elsarticle_v4.tex"
source = tex.read_text(encoding="utf-8")
start = source.index("\\begin{thebibliography}{99}")
end = source.index("\\end{thebibliography}", start)
body = source[start + len("\\begin{thebibliography}{99}"):end]
blocks = [x.strip() for x in re.split(r"(?=\\bibitem\[)", body) if x.strip()]
entries = {}
for block in blocks:
    match = re.match(r"\\bibitem\[[^]]+\]\{([^}]+)\}", block)
    if not match:
        raise ValueError(f"Cannot parse bibliography entry: {block[:80]}")
    entries[match.group(1)] = block

seen = []
for match in re.finditer(r"\\(?:citep|citet|cite)\{([^}]*)\}", source[:start]):
    for key in match.group(1).split(","):
        key = key.strip()
        if key not in seen:
            seen.append(key)
if set(seen) != set(entries):
    raise ValueError(f"Missing entries: {set(seen)-set(entries)}; uncited: {set(entries)-set(seen)}")

ordered = "\n\n" + "\n\n".join(entries[key] for key in seen) + "\n\n"
tex.write_text(source[:start + len("\\begin{thebibliography}{99}")] + ordered + source[end:], encoding="utf-8")
print(f"Ordered {len(seen)} references by first citation: {', '.join(seen)}")
