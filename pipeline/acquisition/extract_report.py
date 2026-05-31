"""
Scheldt Report Extractor — Automated PDF Data Extraction
=========================================================
Two-pass Claude API extraction — splits schema to stay within token limits.

Usage
-----
    pip install anthropic pypdf
    export ANTHROPIC_API_KEY="your-key"
    python extract_report.py report.pdf
    python extract_report.py report.pdf --dry-run
"""

import os, sys, json, csv, argparse
from pathlib import Path
from datetime import datetime

try:
    import anthropic
except ImportError:
    sys.exit("pip install anthropic")

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("pip install pypdf")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL   = "claude-sonnet-4-6"
OUT_DIR = Path("./reports/extracted")

MACROCEL_COORDS = {
    "1": {"lon": 3.55, "lat": 51.44},
    "2": {"lon": 3.70, "lat": 51.39},
    "3": {"lon": 3.85, "lat": 51.38},
    "4": {"lon": 4.00, "lat": 51.38},
    "5": {"lon": 4.10, "lat": 51.37},
    "6": {"lon": 4.20, "lat": 51.37},
    "7": {"lon": 4.28, "lat": 51.37},
}

SYSTEM = """You extract structured data from Scheldt estuary management reports.
Respond with valid JSON only — no markdown fences, no preamble, no explanation.
Translate Dutch field descriptions to English. Use null for missing values."""

# ── TWO EXTRACTION PROMPTS ────────────────────────────────────────────────────

P1 = """Extract from this Scheldt report. JSON only.
{
  "report_metadata": {"title":"","year":0,"period_covered":"","author":"","client":"","language":""},
  "dredging_volumes": [
    {"macrocel":"","location_type":"hoofdgeul|nevengeul|plaatrand|haven",
     "location_name":null,"year":0,"volume_m3":0,"volume_type":"in_situ","notes":null}
  ],
  "disposal_sites": [
    {"name":"","code":null,"macrocel":"","type":"",
     "lon_approx":null,"lat_approx":null,"volume_disposed_m3":null,"year":null,
     "stability_status":"stable|unstable|restricted|monitored",
     "stability_notes":null}
  ],
  "problem_sites": [
    {"name":"","macrocel":null,
     "issue_type":"erosion|sedimentation|instability|restricted_disposal|morphological_change",
     "description":"English description","severity":"high|medium|low",
     "lon_approx":null,"lat_approx":null,"year_identified":null}
  ]
}"""

P2 = """Extract from this Scheldt report. JSON only.
{
  "morphological_data": {
    "intertidal_area_ha": {"year":0,"total_ha":0,"change_from_baseline_ha":null,"baseline_year":null},
    "plate_heights": [{"location":"","year":0,"height_m_NAP":null,"trend":"rising|stable|falling"}]
  },
  "tidal_data": {
    "stations": [
      {"name":"","MHW_m_NAP":null,"MLW_m_NAP":null,"tidal_range_m":null,
       "year":null,"HW_trend_cm_per_year":null,"LW_trend_cm_per_year":null}
    ]
  },
  "validation_sites": [
    {"name":"","type":"known_erosion|known_deposition|stable_reference|monitoring_point",
     "description":"English","lon_approx":null,"lat_approx":null,"macrocel":null}
  ],
  "key_findings": ["one sentence per finding in English"],
  "extraction_notes": ""
}"""

# ── PDF TEXT EXTRACTION ───────────────────────────────────────────────────────

def read_pdf(path: Path, max_pages: int = 100) -> str:
    reader = PdfReader(str(path))
    total  = len(reader.pages)
    print(f"  {total} pages — extracting up to {max_pages}")
    ranges = [(0,5),(15,60),(60,120),(max(0,total-10),total)]
    text   = ""; seen = set()
    for s, e in ranges:
        for i in range(s, min(e, total)):
            if i not in seen and len(seen) < max_pages:
                try:
                    t = reader.pages[i].extract_text()
                    if t: text += f"\n--- PAGE {i+1} ---\n{t}"
                    seen.add(i)
                except Exception:
                    pass
    print(f"  {len(seen)} pages → {len(text):,} chars")
    return text

# ── SINGLE API CALL WITH REPAIR ───────────────────────────────────────────────

def call(client, text: str, prompt: str, n: int) -> dict:
    print(f"  Pass {n}/2 ...", end=" ", flush=True)
    msg = client.messages.create(
        model=MODEL, max_tokens=8000,
        system=SYSTEM,
        messages=[{"role":"user",
                   "content":f"Report text:\n{text[:70000]}\n\n{prompt}"}]
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```",2)[1]
        if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()
    try:
        d = json.loads(raw); print("✓"); return d
    except json.JSONDecodeError:
        for fix in [']}]}]}','"]}]}','"]}','}}','}']:
            try:
                d = json.loads(raw+fix)
                print("✓ (repaired)"); d["_repaired"]=True; return d
            except Exception: pass
        print("✗")
        print(f"    Truncated at char {len(raw)} — first 300: {raw[:300]}")
        return {}

# ── MAIN EXTRACTION ───────────────────────────────────────────────────────────

def extract(path: Path, dry_run: bool = False) -> dict:
    text = read_pdf(path)
    if dry_run:
        print("  [DRY RUN]"); return {"dry_run":True}
    client = anthropic.Anthropic()
    d1 = call(client, text, P1, 1)
    d2 = call(client, text, P2, 2)
    merged = {**d1, **d2}
    # Enrich missing coordinates from macrocel
    for key in ["disposal_sites","problem_sites","validation_sites"]:
        for s in merged.get(key, []):
            if s.get("lon_approx") is None:
                mc = str(s.get("macrocel",""))
                if mc in MACROCEL_COORDS:
                    s["lon_approx"] = MACROCEL_COORDS[mc]["lon"]
                    s["lat_approx"] = MACROCEL_COORDS[mc]["lat"]
                    s["coord_source"] = "macrocel_centre"
    return merged

# ── SAVE ─────────────────────────────────────────────────────────────────────

def save(data: dict, path: Path):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    data["_source"] = path.name
    data["_timestamp"] = datetime.now().isoformat()

    # JSON
    jp = OUT_DIR/f"{stem}_extracted.json"
    with open(jp,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {jp.name}")

    # Markdown summary
    mp = OUT_DIR/f"{stem}_summary.md"
    with open(mp,"w",encoding="utf-8") as f:
        f.write(build_md(data, path))
    print(f"  ✓ {mp.name}")

    # Validation CSV
    sites = data.get("validation_sites",[]) + data.get("problem_sites",[])
    if sites:
        vp = OUT_DIR/f"{stem}_validation.csv"
        fields = ["name","type","issue_type","description",
                  "severity","macrocel","lon_approx","lat_approx",
                  "year_identified"]
        with open(vp,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(sites)
        print(f"  ✓ {vp.name}  ({len(sites)} sites)")

    # Dredging CSV
    drg = data.get("dredging_volumes",[])
    if drg:
        dp = OUT_DIR/f"{stem}_dredging.csv"
        fields = ["macrocel","location_type","location_name",
                  "year","volume_m3","volume_type","notes"]
        with open(dp,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(drg)
        print(f"  ✓ {dp.name}  ({len(drg)} records)")

def build_md(d: dict, path: Path) -> str:
    m = d.get("report_metadata",{})
    md  = f"# {m.get('title', path.name)}\n\n"
    md += f"**Period:** {m.get('period_covered','')}  \n"
    md += f"**Author:** {m.get('author','')}  \n"
    md += f"**Extracted:** {datetime.now():%Y-%m-%d %H:%M}\n\n"
    # Key findings
    for f in d.get("key_findings",[]):
        md += f"- {f}\n"
    md += "\n"
    # Problem sites
    ps = d.get("problem_sites",[])
    if ps:
        md += "## Problem Sites\n| Name | Macrocel | Issue | Severity |\n|---|---|---|---|\n"
        for p in ps:
            md += f"| {p.get('name','')} | {p.get('macrocel','')} | {p.get('issue_type','')} | {p.get('severity','')} |\n"
        md += "\n"
    # Tidal stations
    ts = d.get("tidal_data",{}).get("stations",[])
    if ts:
        md += "## Tidal Stations\n| Station | Year | MHW | MLW | Range |\n|---|---|---|---|---|\n"
        for s in ts:
            md += f"| {s.get('name','')} | {s.get('year','')} | {s.get('MHW_m_NAP','')} | {s.get('MLW_m_NAP','')} | {s.get('tidal_range_m','')} |\n"
    return md

# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pdfs = []
    for p in args.pdfs:
        pdfs += sorted(p.glob("*.pdf")) if p.is_dir() else [p]

    for pdf in pdfs:
        print(f"\n{'='*55}\nProcessing: {pdf.name}\n{'='*55}")
        print("\n[1] Reading PDF ...")
        data = extract(pdf, dry_run=args.dry_run)
        if args.dry_run: continue
        print("\n[2] Saving outputs ...")
        save(data, pdf)
        # Summary
        print(f"\n── Summary ────────────────────────────────────────")
        print(f"  Dredging records : {len(data.get('dredging_volumes',[]))}")
        print(f"  Disposal sites   : {len(data.get('disposal_sites',[]))}")
        print(f"  Problem sites    : {len(data.get('problem_sites',[]))}")
        print(f"  Validation sites : {len(data.get('validation_sites',[]))}")
        print(f"  Tidal stations   : {len(data.get('tidal_data',{}).get('stations',[]))}")
        for f in data.get("key_findings",[])[:3]:
            print(f"  • {f}")
    print(f"\n✓ Done. Outputs in {OUT_DIR}/")
