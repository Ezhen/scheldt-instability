"""
Scheldt Estuary — Monitoring Database Builder
==============================================
Pulls five physical parameters from the Rijkswaterstaat WaterWebServices
API for all major Scheldt stations and stores them in a local SQLite
database with a clean relational schema.

Demonstrates SQL skills:
    - Multi-table relational schema design
    - INSERT with conflict handling (idempotent)
    - JOIN queries across environmental parameters
    - Aggregation, filtering, spatial context

Parameters collected
--------------------
    WATHTE      Water level (cm NAP → m NAP)
    CONCTTE     Suspended sediment concentration (mg/l)
    SALNTT      Salinity (PSU)
    STROOMSHD   Current velocity (m/s)
    WATTEM      Water temperature (°C)

Stations (Western Scheldt + Sea Scheldt)
-----------------------------------------
    VLISSGN     Vlissingen        (mouth)
    TERNZN      Terneuzen
    HANSWT      Hansweert
    BATH        Bath              (near Saeftinghe)
    PROSPPR     Prosperpolder
    KALLSLZS    Kallosluis
    ANTWERPEN   Antwerpen-Loodsgebouw (upstream limit)

Schema
------
    stations        (id, code, name, lon, lat, type)
    parameters      (id, code, name, unit, description)
    observations    (id, station_id, parameter_id, datetime, value, quality)
    risk_sites      (id, name, lon, lat, nearest_station, risk_class,
                     retreat_rate_m_per_yr, issue_type)
    ssc_risk_summary (VIEW — JOIN observations + risk_sites)

Usage
-----
    pip install ddlpy pandas sqlite3
    python acquire_monitoring_db.py

    # Pull only specific parameters
    python acquire_monitoring_db.py --params WATHTE CONCTTE

    # Pull shorter period (faster for testing)
    python acquire_monitoring_db.py --start 2020-01-01 --end 2022-12-31

    # Run example SQL queries after building
    python acquire_monitoring_db.py --query-only
"""

import os
import sys
import json
import sqlite3
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import ddlpy
except ImportError:
    sys.exit("pip install ddlpy")


# ── CONFIG ────────────────────────────────────────────────────────────────────

START_DATE = "2017-01-01"
END_DATE   = "2025-12-31"
DB_PATH    = Path("./data/scheldt/monitoring.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Station metadata (WGS84 coordinates)
STATIONS = {
    "VLISSGN":   {"name": "Vlissingen",             "lon": 3.596,  "lat": 51.442, "type": "tidal"},
    "TERNZN":    {"name": "Terneuzen",               "lon": 3.830,  "lat": 51.344, "type": "tidal"},
    "HANSWT":    {"name": "Hansweert",               "lon": 3.996,  "lat": 51.443, "type": "tidal"},
    "BATH":      {"name": "Bath",                    "lon": 4.212,  "lat": 51.397, "type": "tidal"},
    "PROSPPR":   {"name": "Prosperpolder",           "lon": 4.241,  "lat": 51.368, "type": "tidal"},
    "KALLSLZS":  {"name": "Kallosluis",              "lon": 4.291,  "lat": 51.267, "type": "tidal"},
    "ANTWERPEN": {"name": "Antwerpen-Loodsgebouw",   "lon": 4.396,  "lat": 51.229, "type": "tidal"},
}

# Parameter metadata
PARAMETERS = {
    "WATHTE":    {"name": "Water level",                  "unit": "m NAP",  "scale": 0.01},
    "CONCTTE":   {"name": "Suspended sediment conc.",     "unit": "mg/l",   "scale": 1.0},
    "SALNTT":    {"name": "Salinity",                     "unit": "PSU",    "scale": 1.0},
    "STROOMSHD": {"name": "Current velocity",             "unit": "m/s",    "scale": 1.0},
    "WATTEM":    {"name": "Water temperature",            "unit": "°C",     "scale": 1.0},
}

# Risk sites from validation — nearest RWS station for JOIN queries
RISK_SITES = [
    {"name": "Land van Saeftinghe",          "lon": 4.17,  "lat": 51.36,
     "nearest_station": "BATH",     "risk_class": 2,
     "issue_type": "erosion",        "severity": "medium"},
    {"name": "Biezelingse Ham",              "lon": 3.92,  "lat": 51.44,
     "nearest_station": "HANSWT",   "risk_class": 3,
     "issue_type": "erosion",        "severity": "medium"},
    {"name": "Baarland",                     "lon": 3.89,  "lat": 51.38,
     "nearest_station": "HANSWT",   "risk_class": 2,
     "issue_type": "erosion",        "severity": "low"},
    {"name": "Plaat van Walsoorden",         "lon": 4.02,  "lat": 51.38,
     "nearest_station": "HANSWT",   "risk_class": 2,
     "issue_type": "morphological_change", "severity": "medium"},
    {"name": "Rug van Baarland",             "lon": 3.88,  "lat": 51.40,
     "nearest_station": "HANSWT",   "risk_class": 1,
     "issue_type": "morphological_change", "severity": "medium"},
    {"name": "Geulwandverdediging Saeftinghe","lon": 4.19, "lat": 51.37,
     "nearest_station": "BATH",     "risk_class": 1,
     "issue_type": "engineered_protection", "severity": "low"},
    {"name": "Nauw van Bath",                "lon": 4.21,  "lat": 51.39,
     "nearest_station": "BATH",     "risk_class": 3,
     "issue_type": "navigation_dredging", "severity": "high"},
    {"name": "Plaat van Saeftinghe",         "lon": 4.18,  "lat": 51.38,
     "nearest_station": "BATH",     "risk_class": 2,
     "issue_type": "erosion",       "severity": "medium"},
]


# ── SCHEMA ────────────────────────────────────────────────────────────────────

SCHEMA = """
-- Monitoring stations along the Western Scheldt
CREATE TABLE IF NOT EXISTS stations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    code    TEXT    NOT NULL UNIQUE,
    name    TEXT    NOT NULL,
    lon     REAL    NOT NULL,
    lat     REAL    NOT NULL,
    type    TEXT    DEFAULT 'tidal'
);

-- Physical parameters measured at stations
CREATE TABLE IF NOT EXISTS parameters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    unit        TEXT    NOT NULL,
    description TEXT
);

-- Observations: one row per station × parameter × timestamp
CREATE TABLE IF NOT EXISTS observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id   INTEGER NOT NULL REFERENCES stations(id),
    parameter_id INTEGER NOT NULL REFERENCES parameters(id),
    datetime     TEXT    NOT NULL,
    value        REAL,
    quality      TEXT    DEFAULT 'measured',
    UNIQUE(station_id, parameter_id, datetime)
);

-- Satellite-derived risk sites with nearest monitoring station
CREATE TABLE IF NOT EXISTS risk_sites (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT    NOT NULL UNIQUE,
    lon                    REAL    NOT NULL,
    lat                    REAL    NOT NULL,
    nearest_station        TEXT    REFERENCES stations(code),
    risk_class             INTEGER,
    risk_class_name        TEXT,
    retreat_rate_m_per_yr  REAL,
    issue_type             TEXT,
    severity               TEXT
);

-- Indexes for fast time-range queries
CREATE INDEX IF NOT EXISTS idx_obs_station   ON observations(station_id);
CREATE INDEX IF NOT EXISTS idx_obs_parameter ON observations(parameter_id);
CREATE INDEX IF NOT EXISTS idx_obs_datetime  ON observations(datetime);
CREATE INDEX IF NOT EXISTS idx_obs_combo     ON observations(station_id, parameter_id, datetime);
"""

VIEW_SQL = """
-- View: suspended sediment at risk site nearest stations
-- Answers: are high-SSC events correlated with high-risk zones?
CREATE VIEW IF NOT EXISTS ssc_risk_summary AS
SELECT
    rs.name                          AS site_name,
    rs.risk_class,
    rs.risk_class_name,
    rs.issue_type,
    rs.severity,
    st.code                          AS station,
    st.name                          AS station_name,
    ROUND(AVG(o.value), 2)           AS mean_ssc_mg_l,
    ROUND(MAX(o.value), 2)           AS max_ssc_mg_l,
    ROUND(MIN(o.value), 2)           AS min_ssc_mg_l,
    COUNT(o.id)                      AS n_obs,
    MIN(o.datetime)                  AS period_start,
    MAX(o.datetime)                  AS period_end
FROM risk_sites rs
JOIN stations   st ON st.code = rs.nearest_station
JOIN observations o ON o.station_id = st.id
JOIN parameters  p  ON p.id = o.parameter_id
WHERE p.code = 'CONCTTE'
GROUP BY rs.name, st.code;

-- View: monthly aggregates per station per parameter
CREATE VIEW IF NOT EXISTS monthly_means AS
SELECT
    st.code                          AS station,
    p.code                           AS parameter,
    p.unit,
    SUBSTR(o.datetime, 1, 7)         AS year_month,
    ROUND(AVG(o.value), 4)           AS mean_value,
    ROUND(MIN(o.value), 4)           AS min_value,
    ROUND(MAX(o.value), 4)           AS max_value,
    COUNT(o.id)                      AS n_obs
FROM observations o
JOIN stations   st ON st.id = o.station_id
JOIN parameters p  ON p.id  = o.parameter_id
GROUP BY st.code, p.code, year_month;

-- View: annual trends per station (for detecting long-term change)
CREATE VIEW IF NOT EXISTS annual_trends AS
SELECT
    st.code                          AS station,
    p.code                           AS parameter,
    p.unit,
    SUBSTR(o.datetime, 1, 4)         AS year,
    ROUND(AVG(o.value), 4)           AS annual_mean,
    COUNT(o.id)                      AS n_obs
FROM observations o
JOIN stations   st ON st.id = o.station_id
JOIN parameters p  ON p.id  = o.parameter_id
GROUP BY st.code, p.code, year;
"""


# ── DATABASE SETUP ────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection):
    """Create schema and seed static tables."""
    conn.executescript(SCHEMA)

    # Stations
    for code, meta in STATIONS.items():
        conn.execute("""
            INSERT OR IGNORE INTO stations (code, name, lon, lat, type)
            VALUES (?, ?, ?, ?, ?)
        """, (code, meta["name"], meta["lon"], meta["lat"], meta["type"]))

    # Parameters
    for code, meta in PARAMETERS.items():
        conn.execute("""
            INSERT OR IGNORE INTO parameters (code, name, unit)
            VALUES (?, ?, ?)
        """, (code, meta["name"], meta["unit"]))

    # Risk sites
    cls_names = {1:"Low", 2:"Medium", 3:"High", 4:"Critical"}
    for site in RISK_SITES:
        conn.execute("""
            INSERT OR IGNORE INTO risk_sites
                (name, lon, lat, nearest_station, risk_class,
                 risk_class_name, issue_type, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (site["name"], site["lon"], site["lat"],
              site["nearest_station"], site.get("risk_class"),
              cls_names.get(site.get("risk_class",""), "Unknown"),
              site.get("issue_type"), site.get("severity")))

    conn.commit()
    print("  ✓ Schema created and static tables seeded")


def add_views(conn: sqlite3.Connection):
    """Add analytical views — DROP first to allow recreation."""
    for view in ["ssc_risk_summary", "monthly_means", "annual_trends"]:
        conn.execute(f"DROP VIEW IF EXISTS {view}")
    conn.executescript(VIEW_SQL)
    conn.commit()
    print("  ✓ Analytical views created")


# ── DATA DOWNLOAD ─────────────────────────────────────────────────────────────

def get_station_row(conn, code):
    return conn.execute(
        "SELECT id FROM stations WHERE code = ?", (code,)
    ).fetchone()


def get_param_row(conn, code):
    return conn.execute(
        "SELECT id FROM parameters WHERE code = ?", (code,)
    ).fetchone()


def download_parameter(conn: sqlite3.Connection,
                        param_code: str,
                        station_codes: list[str],
                        start: str, end: str) -> int:
    """
    Download one parameter for all stations and INSERT into observations.
    Returns number of rows inserted.
    """
    print(f"\n  Parameter: {param_code} "
          f"({PARAMETERS[param_code]['name']}) ...")

    # Find this parameter in RWS catalogue
    try:
        locations = ddlpy.locations()
        mask = locations["Grootheid.Code"] == param_code
        available = locations[mask]
        if available.empty:
            print(f"    ✗ {param_code} not in RWS catalogue — skipping")
            return 0
    except Exception as e:
        print(f"    ✗ Catalogue query failed: {e}")
        return 0

    total_inserted = 0
    scale = PARAMETERS[param_code]["scale"]
    param_id = get_param_row(conn, param_code)[0]

    for station_code in station_codes:
        station_row = get_station_row(conn, station_code)
        if not station_row:
            continue
        station_id = station_row[0]

        # Find this station in the available locations
        station_mask = None
        for idx_attempt in [
            available.index == station_code,
        ]:
            try:
                hits = available[idx_attempt]
                if not hits.empty:
                    station_mask = hits
                    break
            except Exception:
                pass

        # Fallback: search by name
        if station_mask is None or (hasattr(station_mask, 'empty') and station_mask.empty):
            if "Naam" in available.columns:
                name = STATIONS.get(station_code, {}).get("name", "")
                hits = available[
                    available["Naam"].str.contains(
                        name.split()[0], case=False, na=False)
                ]
                station_mask = hits if not hits.empty else None

        if station_mask is None or station_mask.empty:
            print(f"    ✗ {station_code}: not found for {param_code}")
            continue

        # Check existing data to avoid re-downloading
        existing = conn.execute("""
            SELECT COUNT(*) FROM observations
            WHERE station_id=? AND parameter_id=?
        """, (station_id, param_id)).fetchone()[0]

        if existing > 100:
            print(f"    ✓ {station_code}: {existing:,} rows already in DB — skip")
            total_inserted += existing
            continue

        # Download
        try:
            row = station_mask.iloc[0]
            df  = ddlpy.measurements(
                row,
                pd.Timestamp(start),
                pd.Timestamp(end)
            )

            if df.empty:
                print(f"    ✗ {station_code}: no data returned")
                continue

            # Find value column
            val_col = None
            for c in ["Meetwaarde.Waarde_Numeriek", "Waarde", "value"]:
                if c in df.columns:
                    val_col = c; break
            if val_col is None:
                num_cols = df.select_dtypes(include="number").columns
                val_col  = num_cols[0] if len(num_cols) > 0 else None
            if val_col is None:
                print(f"    ✗ {station_code}: no numeric column found")
                continue

            # Prepare rows
            values_raw = df[val_col].values.astype(float) * scale
            datetimes  = df.index.astype(str).tolist()

            # Mask outliers
            valid = np.isfinite(values_raw) & (np.abs(values_raw) < 1e6)
            values_raw = np.where(valid, values_raw, np.nan)

            rows = [
                (station_id, param_id, dt, float(v) if np.isfinite(v) else None)
                for dt, v in zip(datetimes, values_raw)
            ]

            conn.executemany("""
                INSERT OR IGNORE INTO observations
                    (station_id, parameter_id, datetime, value)
                VALUES (?, ?, ?, ?)
            """, rows)
            conn.commit()

            n = len(rows)
            total_inserted += n
            print(f"    ✓ {station_code}: {n:,} observations inserted  "
                  f"({start[:4]}–{end[:4]})")

        except Exception as e:
            print(f"    ✗ {station_code}: {e.__class__.__name__}: {e}")

    return total_inserted


# ── EXAMPLE SQL QUERIES ───────────────────────────────────────────────────────

EXAMPLE_QUERIES = {
    "1_ssc_by_risk_class": """
-- Q1: Mean suspended sediment concentration at risk sites
-- Grouped by satellite risk class
-- Tests: do high-risk zones have higher SSC?
SELECT
    risk_class_name,
    COUNT(DISTINCT site_name)  AS n_sites,
    ROUND(AVG(mean_ssc_mg_l),1) AS avg_ssc,
    ROUND(MAX(max_ssc_mg_l),1)  AS peak_ssc
FROM ssc_risk_summary
GROUP BY risk_class_name
ORDER BY risk_class DESC;
""",

    "2_longitudinal_ssc_gradient": """
-- Q2: SSC gradient along the estuary (mouth → upstream)
-- Increasing gradient = net erosion source upstream
SELECT
    st.name   AS station,
    st.lon,
    ROUND(AVG(o.value),2)   AS mean_ssc_mg_l,
    ROUND(MAX(o.value),2)   AS max_ssc_mg_l,
    COUNT(o.id)              AS n_obs
FROM observations o
JOIN stations   st ON st.id = o.station_id
JOIN parameters p  ON p.id  = o.parameter_id
WHERE p.code = 'CONCTTE'
GROUP BY st.code
ORDER BY st.lon ASC;
""",

    "3_saeftinghe_ssc_trend": """
-- Q3: Annual SSC trend at Bath (nearest to Saeftinghe)
-- Rising trend = increasing erosion / changing transport
SELECT
    year,
    ROUND(annual_mean,2)  AS ssc_mg_l,
    n_obs
FROM annual_trends
WHERE station = 'BATH'
  AND parameter = 'CONCTTE'
ORDER BY year;
""",

    "4_high_ssc_events": """
-- Q4: High-turbidity events (SSC > 100 mg/l) per station
-- These are the events that drive cliff erosion
SELECT
    st.name                  AS station,
    SUBSTR(o.datetime,1,7)   AS year_month,
    COUNT(o.id)              AS n_high_turbidity_obs,
    ROUND(MAX(o.value),1)    AS peak_ssc_mg_l
FROM observations o
JOIN stations   st ON st.id = o.station_id
JOIN parameters p  ON p.id  = o.parameter_id
WHERE p.code    = 'CONCTTE'
  AND o.value   > 100
GROUP BY st.code, year_month
ORDER BY n_high_turbidity_obs DESC
LIMIT 20;
""",

    "5_multi_parameter_risk_sites": """
-- Q5: Multi-parameter profile at risk sites
-- JOINs water level, SSC, and salinity at each site's nearest station
-- This is the portfolio SQL demonstration query
SELECT
    rs.name                          AS site,
    rs.risk_class_name               AS risk_class,
    rs.issue_type,
    st.code                          AS station,
    p.code                           AS parameter,
    p.unit,
    ROUND(AVG(o.value), 3)           AS mean_value,
    COUNT(o.id)                      AS n_obs
FROM risk_sites  rs
JOIN stations    st ON st.code = rs.nearest_station
JOIN observations o ON o.station_id = st.id
JOIN parameters  p  ON p.id = o.parameter_id
WHERE p.code IN ('WATHTE', 'CONCTTE', 'SALNTT')
GROUP BY rs.name, p.code
ORDER BY rs.risk_class DESC, rs.name, p.code;
""",

    "6_velocity_vs_ssc": """
-- Q6: Current velocity vs SSC correlation proxy
-- High velocity + high SSC = active sediment transport = erosion risk
SELECT
    st.code                          AS station,
    ROUND(AVG(CASE WHEN p.code='STROOMSHD' THEN o.value END), 3)
                                     AS mean_velocity_ms,
    ROUND(AVG(CASE WHEN p.code='CONCTTE'   THEN o.value END), 2)
                                     AS mean_ssc_mg_l,
    COUNT(DISTINCT
        CASE WHEN p.code='CONCTTE' THEN o.datetime END)
                                     AS n_ssc_obs
FROM observations o
JOIN stations    st ON st.id = o.station_id
JOIN parameters  p  ON p.id  = o.parameter_id
WHERE p.code IN ('STROOMSHD', 'CONCTTE')
GROUP BY st.code
HAVING mean_ssc_mg_l IS NOT NULL
ORDER BY mean_ssc_mg_l DESC NULLS LAST;
""",
}


def run_example_queries(conn: sqlite3.Connection):
    """Run and print all example SQL queries."""
    print("\n" + "="*60)
    print("EXAMPLE SQL QUERIES")
    print("="*60)

    for name, sql in EXAMPLE_QUERIES.items():
        print(f"\n── {name.replace('_',' ').title()} ─────────────────────────")
        print(sql.strip())
        try:
            df = pd.read_sql_query(sql, conn)
            if df.empty:
                print("  (no results — data may not be loaded yet)")
            else:
                print(df.to_string(index=False))
        except Exception as e:
            print(f"  Error: {e}")


def print_db_stats(conn: sqlite3.Connection):
    """Print database statistics."""
    print("\n── Database statistics ────────────────────────────────")
    for table in ["stations", "parameters", "observations", "risk_sites"]:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:20s}: {n:,} rows")

    # Observations by parameter
    print("\n  Observations by parameter:")
    rows = conn.execute("""
        SELECT p.code, p.name, p.unit, COUNT(o.id) AS n
        FROM observations o
        JOIN parameters p ON p.id = o.parameter_id
        GROUP BY p.code
        ORDER BY n DESC
    """).fetchall()
    for code, name, unit, n in rows:
        print(f"    {code:12s} {name:35s} {n:>10,} obs  [{unit}]")

    # Date range
    row = conn.execute("""
        SELECT MIN(datetime), MAX(datetime) FROM observations
    """).fetchone()
    if row[0]:
        print(f"\n  Date range: {row[0][:10]} → {row[1][:10]}")

    # Save schema as SQL file for documentation
    schema_path = DB_PATH.parent / "monitoring_schema.sql"
    with open(schema_path, "w") as f:
        for line in conn.iterdump():
            if line.startswith("CREATE"):
                f.write(line + "\n\n")
    print(f"\n  Schema saved → {schema_path.name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build Scheldt monitoring SQLite database"
    )
    ap.add_argument("--params", nargs="+",
                    default=list(PARAMETERS.keys()),
                    choices=list(PARAMETERS.keys()),
                    help="Parameters to download (default: all)")
    ap.add_argument("--stations", nargs="+",
                    default=list(STATIONS.keys()),
                    choices=list(STATIONS.keys()),
                    help="Stations to download (default: all)")
    ap.add_argument("--start", default=START_DATE,
                    help=f"Start date (default: {START_DATE})")
    ap.add_argument("--end",   default=END_DATE,
                    help=f"End date (default: {END_DATE})")
    ap.add_argument("--query-only", action="store_true",
                    help="Skip download, run example queries only")
    ap.add_argument("--fast", action="store_true",
                    help="Download only WATHTE + CONCTTE for 2020-2022 (quick test)")
    args = ap.parse_args()

    if args.fast:
        args.params   = ["WATHTE", "CONCTTE"]
        args.start    = "2020-01-01"
        args.end      = "2022-12-31"
        args.stations = ["VLISSGN", "HANSWT", "BATH"]

    print("=== Scheldt Monitoring Database ===")
    print(f"  DB path    : {DB_PATH}")
    print(f"  Parameters : {args.params}")
    print(f"  Stations   : {args.stations}")
    print(f"  Period     : {args.start} → {args.end}\n")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")  # faster concurrent writes

    print("[1] Initialising schema ...")
    init_db(conn)
    add_views(conn)

    if not args.query_only:
        print("\n[2] Downloading observations ...")
        total = 0
        for param in args.params:
            if param not in PARAMETERS:
                print(f"  Skipping unknown parameter: {param}")
                continue
            n = download_parameter(conn, param, args.stations,
                                    args.start, args.end)
            total += n
        print(f"\n  Total observations inserted: {total:,}")

    print("\n[3] Database statistics:")
    print_db_stats(conn)

    print("\n[4] Running example queries ...")
    run_example_queries(conn)

    conn.close()
    print(f"\n✓ Database ready: {DB_PATH}")
    print(f"\nConnect from Python:")
    print(f"  import sqlite3, pandas as pd")
    print(f"  conn = sqlite3.connect('{DB_PATH}')")
    print(f"  df   = pd.read_sql_query('SELECT * FROM ssc_risk_summary', conn)")
    print(f"\nOr explore interactively:")
    print(f"  sqlite3 {DB_PATH}")
    print(f"  .tables")
    print(f"  .schema observations")
