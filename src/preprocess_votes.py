#!/usr/bin/env python3
"""
preprocess_votes.py — One-time script to parse all house/senate vote XMLs
into flat per-member CSVs.

Run once from the src/ directory:
    python preprocess_votes.py

Outputs:
    data/house_votes_members.csv
    data/senate_votes_members.csv
    data/legislators/lis_to_bioguide.csv

Uses multiprocessing for speed. Safe to re-run (overwrites existing CSVs).
"""

import glob
import logging
import multiprocessing as mp
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SRC_DIR  = Path(__file__).resolve().parent
DATA_DIR = SRC_DIR / "data"

HOUSE_VOTES_DIR  = DATA_DIR / "house_votes"
SENATE_VOTES_DIR = DATA_DIR / "senate_votes"
LEGISLATORS_DIR  = DATA_DIR / "legislators"

OUT_HOUSE  = DATA_DIR / "house_votes_members.csv"
OUT_SENATE = DATA_DIR / "senate_votes_members.csv"
OUT_LIS    = LEGISLATORS_DIR / "lis_to_bioguide.csv"

# ---------------------------------------------------------------------------
# Vote cast normalisation
# ---------------------------------------------------------------------------

_VOTE_MAP = {
    "yea":        "Yea",
    "aye":        "Yea",
    "nay":        "Nay",
    "no":         "Nay",
    "present":    "Not Voting",
    "not voting": "Not Voting",
    "not-voting": "Not Voting",
}

def _norm(raw: str) -> str:
    return _VOTE_MAP.get((raw or "").strip().lower(), "Not Voting")


# ---------------------------------------------------------------------------
# lis -> bioguide mapping
# ---------------------------------------------------------------------------

def build_lis_to_bioguide() -> dict:
    """Parse legislators-current + legislators-historical and return lis -> bioguide."""
    all_members = []
    for fname in ("legislators-current.yaml", "legislators-historical.yaml"):
        path = LEGISLATORS_DIR / fname
        if not path.exists():
            logger.warning("Missing: %s", path)
            continue
        with open(path, encoding="utf-8") as fh:
            all_members.extend(yaml.safe_load(fh))

    mapping = {}
    for m in all_members:
        lis      = m.get("id", {}).get("lis", "")
        bioguide = m.get("id", {}).get("bioguide", "")
        if lis and bioguide:
            mapping[lis] = bioguide

    logger.info("lis->bioguide: %d entries", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# House XML parsing  (per-file worker)
# ---------------------------------------------------------------------------

def _parse_house_xml(filepath: str) -> list[dict]:
    try:
        root = ET.parse(filepath).getroot()
    except ET.ParseError as exc:
        logger.warning("Bad XML %s: %s", filepath, exc)
        return []

    meta = root.find("vote-metadata")
    if meta is None:
        return []

    def _t(tag):
        el = meta.find(tag)
        return el.text.strip() if el is not None and el.text else ""

    congress    = _t("congress")
    rollcall    = _t("rollcall-num")
    vote_date   = _t("action-date")
    bill_number = _t("legis-num")
    bill_title  = _t("vote-desc")
    question    = _t("vote-question")
    session     = _t("session")
    vote_id   = f"house_{congress}_{session}_{rollcall}"

    vote_data = root.find("vote-data")
    if vote_data is None:
        return []

    rows = []
    for rv in vote_data.findall("recorded-vote"):
        leg = rv.find("legislator")
        v   = rv.find("vote")
        if leg is None or v is None:
            continue
        rows.append({
            "congress":    congress,
            "chamber":     "house",
            "rollcall_num":rollcall,
            "vote_id":     vote_id,
            "vote_date":   vote_date,
            "bill_number": bill_number,
            "bill_title":  bill_title,
            "question":    question,
            "session":     session,
            "bioguide":    leg.attrib.get("name-id", ""),
            "member_name": leg.attrib.get("unaccented-name", leg.text or "").strip(),
            "party":       leg.attrib.get("party", ""),
            "state":       leg.attrib.get("state", ""),
            "vote_cast":   _norm(v.text or ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Senate XML parsing  (per-file worker)
# ---------------------------------------------------------------------------

def _parse_senate_xml(filepath: str) -> list[dict]:
    try:
        root = ET.parse(filepath).getroot()
    except ET.ParseError as exc:
        logger.warning("Bad XML %s: %s", filepath, exc)
        return []

    def _t(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else ""

    congress  = _t("congress")
    session   = _t("session")
    vote_num  = _t("vote_number")
    vote_id   = f"senate_{congress}_{session}_{vote_num}"
    vote_date = _t("vote_date")
    question  = _t("question")
    bill_title  = _t("vote_title")
    bill_number = _t("document_name") or _t("vote_document_text")[:80]

    rows = []
    for m in root.findall(".//member"):
        lis_id = (m.findtext("lis_member_id") or "").strip()
        rows.append({
            "congress":      congress,
            "chamber":       "senate",
            "rollcall_num":  vote_num,
            "vote_id":       vote_id,
            "vote_date":     vote_date,
            "bill_number":   bill_number,
            "bill_title":    bill_title,
            "question":      question,
            "congress":      congress,
            "session":       session,
            "lis_member_id": lis_id,
            "member_name":   (m.findtext("member_full") or "").strip(),
            "party":         (m.findtext("party") or "").strip(),
            "state":         (m.findtext("state") or "").strip(),
            "vote_cast":     _norm(m.findtext("vote_cast") or ""),
        })
    return rows


def _process_house_chunk(files: list[str]) -> list[dict]:
    rows = []
    for f in files:
        rows.extend(_parse_house_xml(f))
    return rows

def _process_senate_chunk(files: list[str]) -> list[dict]:
    rows = []
    for f in files:
        rows.extend(_parse_senate_xml(f))
    return rows


# ---------------------------------------------------------------------------
# Parallel batch runner
# ---------------------------------------------------------------------------

def _run_parallel(files: list[str], chunk_fn, label: str) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_workers = mp.cpu_count()
    logger.info("Parsing %d %s XMLs with %d workers …", len(files), label, n_workers)

    chunk_size = max(1, len(files) // (n_workers * 4))
    chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]

    rows = []
    completed_files = 0
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(chunk_fn, chunk): len(chunk) for chunk in chunks}
        for fut in as_completed(futures):
            rows.extend(fut.result())
            completed_files += futures[fut]
            logger.info("  %d / %d files …", completed_files, len(files))

    logger.info("  → %d rows from %s XMLs", len(rows), label)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- lis -> bioguide map -----------------------------------------------
    lis_to_bio = build_lis_to_bioguide()
    logging.info("start processing ...")

    # Save as standalone CSV for transparency
    OUT_LIS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        list(lis_to_bio.items()), columns=["lis_member_id", "bioguide"]
    ).to_csv(OUT_LIS, index=False)
    logger.info("Saved %s", OUT_LIS)

    # ---- House votes -------------------------------------------------------
    house_files = glob.glob(str(HOUSE_VOTES_DIR / "**" / "*.xml"), recursive=True)
    if not house_files:
        logger.error("No house XML files found in %s", HOUSE_VOTES_DIR)
        sys.exit(1)

    house_rows = _run_parallel(house_files, _process_house_chunk, "house")
    house_df = pd.DataFrame(house_rows)
    house_df["vote_date"] = pd.to_datetime(
        house_df["vote_date"], format="%d-%b-%Y", errors="coerce"
    )
    house_df.dropna(subset=["vote_date"], inplace=True)
    house_df.sort_values("vote_date", inplace=True)
    house_df.reset_index(drop=True, inplace=True)
    house_df.to_csv(OUT_HOUSE, index=False)
    logger.info("Saved %s  (%d rows, %d unique members, %d unique votes)",
                OUT_HOUSE, len(house_df),
                house_df["bioguide"].nunique(),
                house_df["vote_id"].nunique())

    # ---- Senate votes ------------------------------------------------------
    senate_files = glob.glob(str(SENATE_VOTES_DIR / "**" / "*.xml"), recursive=True)
    if not senate_files:
        logger.error("No senate XML files found in %s", SENATE_VOTES_DIR)
        sys.exit(1)

    senate_rows = _run_parallel(senate_files, _process_senate_chunk, "senate")
    senate_df = pd.DataFrame(senate_rows)

    # Resolve lis_member_id -> bioguide
    senate_df["bioguide"] = senate_df["lis_member_id"].map(lis_to_bio)
    n_unmapped = senate_df["bioguide"].isna().sum()
    if n_unmapped:
        logger.warning(
            "%d senate member-vote rows have no bioguide mapping (%.1f%%).",
            n_unmapped, n_unmapped / len(senate_df) * 100,
        )

    senate_df["vote_date"] = pd.to_datetime(
        senate_df["vote_date"], errors="coerce"
    )
    senate_df.dropna(subset=["vote_date"], inplace=True)
    senate_df.sort_values("vote_date", inplace=True)
    senate_df.reset_index(drop=True, inplace=True)
    senate_df.to_csv(OUT_SENATE, index=False)
    logger.info("Saved %s  (%d rows, %d unique members, %d unique votes)",
                OUT_SENATE, len(senate_df),
                senate_df["bioguide"].nunique(),
                senate_df["vote_id"].nunique())

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
