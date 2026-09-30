"""Read-only DOI metadata check for the references in the v4 TeX source.

Crossref and DataCite metadata can detect an incorrect DOI/title pairing. They
do not establish that a cited result supports the manuscript claim, and they
do not replace author approval of the final bibliography.
"""
from __future__ import annotations

import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent.parent / "AE_paper_elsarticle_v4.tex"
DOI = re.compile(r"doi\.org/([^}\s]+)", re.I)
ITEM = re.compile(r"\\bibitem\[[^]]*\]\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})", re.S)


def plain(source: str) -> str:
    source = re.sub(r"\\url\{[^}]*\}", "", source)
    source = re.sub(r"\\(?:textit|emph)\{([^}]*)\}", r"\1", source)
    source = re.sub(r"\\[A-Za-z]+", " ", source)
    source = re.sub(r"[{}~]", " ", source)
    return " ".join(source.split())


def similarity(a: str, b: str) -> float:
    clean = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
    return SequenceMatcher(None, clean(a), clean(b)).ratio()


def fetch(doi: str) -> dict:
    api = ("https://api.datacite.org/dois/" if doi.lower().startswith(("10.48550/", "10.5281/", "10.17632/"))
           else "https://api.crossref.org/works/") + urllib.parse.quote(doi, safe="")
    req = urllib.request.Request(api, headers={"User-Agent": "AEpaperReferenceAudit/1.0 (metadata check)"})
    try:
        with urllib.request.urlopen(req, timeout=25) as stream:
            payload = json.load(stream)
        if "data" in payload:
            meta = payload["data"]["attributes"]
            return {"status": "found_datacite", "title": meta.get("titles", [{}])[0].get("title", ""),
                    "year": meta.get("publicationYear", ""), "metadata_url": api,
                    "authors": "; ".join(c.get("name", "") for c in meta.get("creators", [])),
                    "venue": meta.get("publisher", ""), "volume": "", "pages": ""}
        meta = payload["message"]
        published = meta.get("published-print") or meta.get("published-online") or meta.get("issued", {})
        parts = published.get("date-parts", [[""]])
        return {"status": "found_crossref", "title": meta.get("title", [""])[0],
                "year": parts[0][0] if parts else "", "metadata_url": api,
                "authors": "; ".join(" ".join(filter(None, (a.get("given", ""), a.get("family", ""))))
                                     for a in meta.get("author", [])),
                "venue": meta.get("container-title", [""])[0],
                "volume": meta.get("volume", ""), "pages": meta.get("page", meta.get("article-number", ""))}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        return {"status": "lookup_failed", "title": "", "year": "", "authors": "",
                "venue": "", "volume": "", "pages": "",
                "metadata_url": api, "error": str(exc)}


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    records = []
    for key, body in ITEM.findall(tex):
        match = DOI.search(body)
        doi = match.group(1).rstrip(".,; ") if match else ""
        cited_title = ""
        if doi:
            before_url = body[:match.start()]
            # Bibliography uses author/year sentence then title sentence.
            sentences = re.split(r"\(\d{4}\)\.\s*", plain(before_url), maxsplit=1)
            if len(sentences) > 1:
                cited_title = sentences[1].split(".", 1)[0].strip()
        metadata = fetch(doi) if doi else {"status": "no_doi", "title": "", "year": "",
                                       "authors": "", "venue": "", "volume": "", "pages": "",
                                       "metadata_url": ""}
        agreement = similarity(cited_title, metadata["title"]) if cited_title and metadata["title"] else ""
        records.append({"bibkey": key, "doi": doi, "cited_title": cited_title,
                        "registered_title": metadata["title"], "registered_year": metadata["year"],
                        "registered_authors": metadata["authors"], "registered_venue": metadata["venue"],
                        "registered_volume": metadata["volume"], "registered_pages": metadata["pages"],
                        "title_similarity": agreement, "lookup_status": metadata["status"],
                        "metadata_url": metadata["metadata_url"], "lookup_error": metadata.get("error", "")})
        print(f"{key}: {metadata['status']} similarity={agreement}", flush=True)
        time.sleep(0.12)
    with (HERE / "reference_metadata_audit.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
