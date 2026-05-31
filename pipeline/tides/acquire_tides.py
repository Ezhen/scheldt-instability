import os
os.environ["CPL_LOG"] = "/dev/null"

"""
Scheldt Estuary — Vlissingen Tide Gauge Data
=============================================
Downloads water level observations from Rijkswaterstaat DDL API
for station Vlissingen (VLISSGN), 2017–2025.

Computes tidal datums:
    MHW  = Mean High Water
    MLW  = Mean Low Water
    MTL  = Mean Tidal Level  (MHW + MLW) / 2
    MHWS = Mean High Water Springs
    MLWS = Mean Low Water Springs
    TR   = Tidal Range  (MHW - MLW)

Outputs
-------
    tides/vlissingen_hourly.csv      hourly water levels (m NAP)
    tides/vlissingen_extremes.csv    HW/LW peaks per tidal cycle
    tides/tidal_datums.json          MHW, MLW, MTL, TR in metres NAP
    tides/tides_overview.png         time series + distribution figure

No authentication needed — RWS DDL is fully open.

Dependencies
------------
    pip install ddlpy pandas numpy matplotlib scipy

Usage
-----
    python acquire_tides.py
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

try:
    import ddlpy
except ImportError:
    import sys
    sys.exit(
        "ddlpy not installed.\n"
        "Run: pip install ddlpy\n"
        "Then retry."
    )

# ── CONFIG ────────────────────────────────────────────────────────────────────

STATION    = "VLISSGN"      # Vlissingen — western mouth of the Scheldt
START_DATE = "2017-01-01"
END_DATE   = "2025-12-31"

OUT_DIR = Path("./data/scheldt/tides")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HOURLY   = OUT_DIR / "vlissingen_hourly.csv"
OUT_EXTREMES = OUT_DIR / "vlissingen_extremes.csv"
OUT_DATUMS   = OUT_DIR / "tidal_datums.json"
OUT_PNG      = OUT_DIR / "tides_overview.png"


# ── STEP 1: FIND STATION IN CATALOGUE ────────────────────────────────────────

def find_station() -> pd.Series:
    print("[1] Querying RWS catalogue for VLISSGN WATHTE ...")
    locations = ddlpy.locations()
    print(f"  Total locations in catalogue: {len(locations)}")
    print(f"  Index type: {type(locations.index)}")
    print(f"  Index sample: {list(locations.index[:5])}")

    # Try multiple filter approaches — ddlpy API changed between versions
    station = pd.DataFrame()

    # Approach 1: direct index (new ddlpy)
    if STATION in locations.index:
        mask = (
            (locations.index == STATION) &
            (locations["Grootheid.Code"] == "WATHTE")
        )
        station = locations[mask]
        print(f"  Approach 1 (direct index) found {len(station)} rows")

    # Approach 2: MultiIndex with Code level (old ddlpy)
    if station.empty:
        try:
            mask = (
                (locations.index.get_level_values("Code") == STATION) &
                (locations["Grootheid.Code"] == "WATHTE")
            )
            station = locations[mask]
            print(f"  Approach 2 (MultiIndex Code) found {len(station)} rows")
        except Exception:
            pass

    # Approach 3: search by name/code in any column
    if station.empty:
        try:
            for col in ["Code", "Naam"]:
                if col in locations.columns:
                    mask = (
                        locations[col].str.upper().str.contains("VLISS", na=False) &
                        (locations["Grootheid.Code"] == "WATHTE")
                    )
                    station = locations[mask]
                    if not station.empty:
                        print(f"  Approach 3 (column {col}) found {len(station)} rows")
                        break
        except Exception:
            pass

    if station.empty:
        # Print full catalogue of WATHTE stations to help identify correct code
        print(f"\n  Could not find {STATION}.")
        print(f"  All WATHTE stations in catalogue:")
        wathte = locations[locations["Grootheid.Code"] == "WATHTE"]
        print(f"  Index codes: {list(wathte.index[:30])}")
        if "Naam" in wathte.columns:
            print(f"  Names: {list(wathte['Naam'].dropna().unique()[:30])}")
        if "Code" in wathte.columns:
            print(f"  Code col: {list(wathte['Code'].dropna().unique()[:30])}")
        raise ValueError(
            f"Station {STATION} not found. "
            f"Check the printed list above and update STATION variable."
        )

    print(f"  ✓ Found {STATION}: {len(station)} parameter row(s)")
    row = station.iloc[0]
    for field in ["Naam", "X", "Y", "Grootheid.Code", "Eenheid.Code"]:
        if field in row.index:
            print(f"    {field}: {row[field]}")
    return row


# ── STEP 2: DOWNLOAD OBSERVATIONS ────────────────────────────────────────────

def download_observations(station_row: pd.Series) -> pd.DataFrame:
    """
    Download water level observations in annual chunks.
    RWS DDL API max window is ~31 days per request.
    ddlpy handles chunking automatically.
    """
    print(f"\n[2] Downloading water levels {START_DATE} → {END_DATE} ...")
    print(f"  This may take several minutes — API fetches in monthly chunks")

    start = pd.Timestamp(START_DATE)
    end   = pd.Timestamp(END_DATE)

    measurements = ddlpy.measurements(station_row, start_date=start,
                                       end_date=end)

    if measurements.empty:
        raise ValueError("No measurements returned. Check date range.")

    print(f"  ✓ Downloaded {len(measurements):,} observations")
    print(f"    Period  : {measurements.index.min()} → "
          f"{measurements.index.max()}")
    # Column name changed between ddlpy versions
    # Could be 'Waarde', 'value', or similar — detect automatically
    # ddlpy column name changed — actual value is in Meetwaarde.Waarde_Numeriek
    val_col = None
    for candidate in ["Meetwaarde.Waarde_Numeriek", "Waarde",
                       "value", "Value", "waarde"]:
        if candidate in measurements.columns:
            val_col = candidate
            break
    if val_col is None:
        print(f"  Available columns: {list(measurements.columns)}")
        # Find first numeric column
        num_cols = measurements.select_dtypes(include="number").columns
        val_col  = num_cols[0] if len(num_cols) > 0 else measurements.columns[0]
        print(f"  Falling back to: {val_col}")

    print(f"    Columns : {list(measurements.columns[:6])}")
    print(f"    Value col: {val_col}")
    print(f"    Raw range: {measurements[val_col].min():.3f} – "
          f"{measurements[val_col].max():.3f} "
          f"{station_row.get('Eenheid.Code', '?')}")

    # Store detected column name for downstream use
    measurements.attrs["val_col"] = val_col
    return measurements


# ── STEP 3: CLEAN + RESAMPLE TO HOURLY ───────────────────────────────────────

def clean_and_resample(raw: pd.DataFrame,
                        unit: str) -> pd.DataFrame:
    """
    Clean raw observations and resample to hourly.
    RWS DDL returns 10-minute data for Vlissingen.
    Unit is typically 'cm' — convert to metres.
    """
    print("\n[3] Cleaning and resampling to hourly ...")

    val_col = raw.attrs.get("val_col", "Meetwaarde.Waarde_Numeriek")
    if val_col not in raw.columns:
        num_cols = raw.select_dtypes(include="number").columns.tolist()
        val_col  = num_cols[0] if num_cols else raw.columns[0]
        print(f"  Using column: {val_col}")
    df = raw[[val_col]].copy()
    df.columns = ["wl_raw"]

    # Convert cm → m if needed
    if unit.lower() == "cm":
        df["wl_m"] = df["wl_raw"] / 100.0
        print("  ✓ Converted cm → m NAP")
    else:
        df["wl_m"] = df["wl_raw"]
        print(f"  Unit is '{unit}' — assuming metres NAP")

    # Remove obvious outliers (>10m or <-10m NAP for Vlissingen)
    n_before = len(df)
    df = df[(df["wl_m"] > -10) & (df["wl_m"] < 10)]
    print(f"  Outliers removed: {n_before - len(df)}")

    # Resample to hourly mean
    hourly = df["wl_m"].resample("1h").mean()
    hourly = hourly.interpolate(method="time", limit=3)  # fill short gaps
    hourly = hourly.dropna()

    print(f"  ✓ Hourly series: {len(hourly):,} observations")
    print(f"    Range: {hourly.min():.3f} – {hourly.max():.3f} m NAP")
    print(f"    Mean : {hourly.mean():.3f} m NAP")

    return hourly.to_frame(name="wl_m")


# ── STEP 4: DETECT TIDAL EXTREMES ────────────────────────────────────────────

def detect_extremes(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Detect high water (HW) and low water (LW) peaks in the tidal signal.
    Uses scipy.signal.find_peaks on the hourly series.
    Tidal cycle at Vlissingen is ~12.4 hours — min distance = 10 hours.
    """
    print("\n[4] Detecting tidal extremes ...")

    wl = hourly["wl_m"].values
    ts = hourly.index

    # Find high water peaks
    hw_idx, _ = find_peaks(wl, distance=10, prominence=0.3)
    # Find low water troughs (invert signal)
    lw_idx, _ = find_peaks(-wl, distance=10, prominence=0.3)

    hw = pd.DataFrame({
        "datetime": ts[hw_idx],
        "wl_m":     wl[hw_idx],
        "type":     "HW",
    })
    lw = pd.DataFrame({
        "datetime": ts[lw_idx],
        "wl_m":     wl[lw_idx],
        "type":     "LW",
    })

    extremes = pd.concat([hw, lw]).sort_values("datetime").reset_index(drop=True)
    print(f"  ✓ High waters : {len(hw):,}  (mean = {hw['wl_m'].mean():.3f} m NAP)")
    print(f"  ✓ Low waters  : {len(lw):,}  (mean = {lw['wl_m'].mean():.3f} m NAP)")

    return extremes


# ── STEP 5: COMPUTE TIDAL DATUMS ─────────────────────────────────────────────

def compute_datums(hourly: pd.DataFrame,
                   extremes: pd.DataFrame) -> dict:
    """
    Compute standard tidal datums from the full record.
    All values in metres NAP.
    """
    print("\n[5] Computing tidal datums ...")

    hw = extremes[extremes["type"] == "HW"]["wl_m"]
    lw = extremes[extremes["type"] == "LW"]["wl_m"]

    # Standard datums
    MHW  = float(hw.mean())
    MLW  = float(lw.mean())
    MTL  = (MHW + MLW) / 2
    TR   = MHW - MLW

    # Spring tides: top/bottom 10% of HW/LW
    MHWS = float(hw.quantile(0.90))
    MLWS = float(lw.quantile(0.10))
    MHWN = float(hw.quantile(0.10))   # neap HW
    MLWN = float(lw.quantile(0.90))   # neap LW

    # Mean sea level from hourly record
    MSL  = float(hourly["wl_m"].mean())

    datums = {
        "station":    STATION,
        "period":     f"{START_DATE} to {END_DATE}",
        "n_hw":       int(len(hw)),
        "n_lw":       int(len(lw)),
        "MHW":        round(MHW,  4),
        "MLW":        round(MLW,  4),
        "MTL":        round(MTL,  4),
        "TR":         round(TR,   4),
        "MHWS":       round(MHWS, 4),
        "MLWS":       round(MLWS, 4),
        "MHWN":       round(MHWN, 4),
        "MLWN":       round(MLWN, 4),
        "MSL":        round(MSL,  4),
        "units":      "m NAP",
        "note": (
            "MHW/MLW = mean of all HW/LW peaks. "
            "MHWS = 90th percentile HW (spring). "
            "MLWS = 10th percentile LW (spring). "
            "EAD = elevation - MHW."
        ),
    }

    print(f"  MHW  = {MHW:+.3f} m NAP  (mean high water)")
    print(f"  MLW  = {MLW:+.3f} m NAP  (mean low water)")
    print(f"  MTL  = {MTL:+.3f} m NAP  (mean tidal level)")
    print(f"  TR   = {TR:.3f} m        (tidal range)")
    print(f"  MHWS = {MHWS:+.3f} m NAP  (mean high water springs)")
    print(f"  MLWS = {MLWS:+.3f} m NAP  (mean low water springs)")
    print(f"  MSL  = {MSL:+.3f} m NAP  (mean sea level)")

    return datums


# ── STEP 6: VISUALISE ────────────────────────────────────────────────────────

def visualise(hourly: pd.DataFrame,
              extremes: pd.DataFrame,
              datums: dict):
    print("\n[6] Generating figure ...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), facecolor="#0e0e0e")

    # ── Panel 1: Full time series ─────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#111")
    # Annual means to show long-term trend
    try:
        annual = hourly["wl_m"].resample("YE").mean()
    except Exception:
        annual = hourly["wl_m"].resample("A").mean()
    ax.plot(hourly.index, hourly["wl_m"],
            color="#4FC3F7", lw=0.3, alpha=0.4, label="Hourly")
    ax.plot(annual.index, annual.values,
            color="#FF9800", lw=2.0, marker="o",
            markersize=5, label="Annual mean")
    ax.axhline(datums["MHW"],  color="#EF5350", lw=1.2,
               ls="--", label=f"MHW = {datums['MHW']:+.3f} m")
    ax.axhline(datums["MLW"],  color="#42A5F5", lw=1.2,
               ls="--", label=f"MLW = {datums['MLW']:+.3f} m")
    ax.axhline(datums["MTL"],  color="#FFFFFF", lw=0.8,
               ls=":", label=f"MTL = {datums['MTL']:+.3f} m")
    ax.set_ylabel("Water level (m NAP)", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#444")
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white",
              loc="upper left", framealpha=0.9)
    ax.set_title(f"Vlissingen (VLISSGN) — Water level 2017–2025",
                  color="white", fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # ── Panel 2: One spring-neap cycle (2 weeks) ──────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#111")
    # Pick a representative 2-week window — first Jan 2020
    # Hourly index is timezone-aware — slice timestamps must match
    tz = hourly.index.tz
    t0 = pd.Timestamp("2020-01-01", tz=tz)
    t1 = pd.Timestamp("2020-01-15", tz=tz)
    zoom = hourly.loc[t0:t1, "wl_m"]
    ax2.plot(zoom.index, zoom.values, color="#4FC3F7", lw=1.5)
    ax2.fill_between(zoom.index, zoom.values, datums["MTL"],
                      where=zoom.values > datums["MTL"],
                      alpha=0.25, color="#EF5350", label="Above MTL")
    ax2.fill_between(zoom.index, zoom.values, datums["MTL"],
                      where=zoom.values < datums["MTL"],
                      alpha=0.25, color="#42A5F5", label="Below MTL")
    ax2.axhline(datums["MHW"],  color="#EF5350", lw=1.0, ls="--")
    ax2.axhline(datums["MLW"],  color="#42A5F5", lw=1.0, ls="--")
    ax2.axhline(0.0,            color="#FFFFFF",  lw=0.6, ls=":")
    ax2.set_ylabel("Water level (m NAP)", color="white", fontsize=9)
    ax2.set_title("Spring-neap cycle detail (Jan 2020)",
                   color="white", fontsize=10, fontweight="bold")
    ax2.tick_params(colors="white", labelsize=8)
    ax2.spines[:].set_color("#444")
    ax2.legend(fontsize=8, facecolor="#222", labelcolor="white")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))

    # ── Panel 3: HW/LW distribution ───────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#111")
    hw_vals = extremes[extremes["type"] == "HW"]["wl_m"]
    lw_vals = extremes[extremes["type"] == "LW"]["wl_m"]
    ax3.hist(hw_vals, bins=80, color="#EF5350", alpha=0.75,
             label=f"High water  (n={len(hw_vals):,}  μ={hw_vals.mean():.2f}m)")
    ax3.hist(lw_vals, bins=80, color="#42A5F5", alpha=0.75,
             label=f"Low water   (n={len(lw_vals):,}  μ={lw_vals.mean():.2f}m)")
    for val, col, lbl in [
        (datums["MHW"],  "#EF5350", "MHW"),
        (datums["MLW"],  "#42A5F5", "MLW"),
        (datums["MHWS"], "#FF7043", "MHWS"),
        (datums["MLWS"], "#1E88E5", "MLWS"),
    ]:
        ax3.axvline(val, color=col, lw=1.5, ls="--",
                    label=f"{lbl} = {val:+.3f} m")
    ax3.set_xlabel("Water level (m NAP)", color="white", fontsize=9)
    ax3.set_ylabel("Count", color="white", fontsize=9)
    ax3.set_title("HW / LW distribution — tidal datums",
                   color="white", fontsize=10, fontweight="bold")
    ax3.tick_params(colors="white", labelsize=8)
    ax3.spines[:].set_color("#444")
    ax3.legend(fontsize=8, facecolor="#222", labelcolor="white",
               loc="upper left", framealpha=0.9)

    # ── Datum annotation box ──────────────────────────────────────────────────
    txt = (
        f"Tidal datums — Vlissingen (NAP)\n"
        f"MHW  = {datums['MHW']:+.3f} m\n"
        f"MLW  = {datums['MLW']:+.3f} m\n"
        f"TR   = {datums['TR']:.3f} m\n"
        f"MTL  = {datums['MTL']:+.3f} m\n"
        f"MSL  = {datums['MSL']:+.3f} m"
    )
    axes[2].text(0.78, 0.95, txt, transform=axes[2].transAxes,
                  color="white", fontsize=8, va="top",
                  bbox=dict(facecolor="#333", alpha=0.9,
                            boxstyle="round", edgecolor="#555"))

    fig.suptitle("Scheldt — Vlissingen tide gauge  |  Rijkswaterstaat DDL",
                  color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(OUT_PNG), dpi=150, bbox_inches="tight",
                facecolor="#0e0e0e")
    print(f"  → {OUT_PNG.name}")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Vlissingen Tide Gauge — Rijkswaterstaat DDL ===\n")

    station_row  = find_station()
    raw          = download_observations(station_row)
    unit         = station_row.get("Eenheid.Code", "cm")
    hourly       = clean_and_resample(raw, unit)
    extremes     = detect_extremes(hourly)
    datums       = compute_datums(hourly, extremes)

    # Save outputs
    print("\nSaving outputs ...")
    hourly.to_csv(str(OUT_HOURLY))
    print(f"  ✓ {OUT_HOURLY.name}  ({len(hourly):,} rows)")

    extremes.to_csv(str(OUT_EXTREMES), index=False)
    print(f"  ✓ {OUT_EXTREMES.name}  ({len(extremes):,} rows)")

    with open(OUT_DATUMS, "w") as f:
        json.dump(datums, f, indent=2)
    print(f"  ✓ {OUT_DATUMS.name}")

    visualise(hourly, extremes, datums)

    print(f"\n✓ Complete. Tidal datums for EAD computation:")
    print(f"  MHW = {datums['MHW']:+.3f} m NAP")
    print(f"  EAD = AHN4 elevation - {datums['MHW']:+.3f}")
    print(f"\nNext: python compute_ead.py")
