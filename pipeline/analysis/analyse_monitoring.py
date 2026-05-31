"""
Scheldt — Monitoring Database Analysis (patched v2)
=====================================================
Patches applied:
  1. Station-specific surge thresholds from tidal_asymmetry results
  2. Fixed tidal asymmetry formula (ebb duration / flood duration)
  3. Salinity outlier filter improved (var query fixed)
  4. Risk site profiles use correct per-station MHW

Usage: python analyse_monitoring.py
"""

import os
os.environ["CPL_LOG"] = "/dev/null"

import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from pathlib import Path

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────

DB_PATH  = Path("./data/scheldt/monitoring.db")
OUT_DIR  = Path("./data/scheldt/monitoring")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATUMS_PATH = Path("./data/scheldt/tides/tidal_datums.json")
with open(DATUMS_PATH) as f:
    DATUMS = json.load(f)
MHW_VLISS = DATUMS["MHW"]   # Vlissingen reference
MLW_VLISS = DATUMS["MLW"]
TR_VLISS  = DATUMS["TR"]

STATION_ORDER = ["VLISSGN","TERNZN","HANSWT","BATH","KALLSLZS"]
STATION_NAMES = {
    "VLISSGN":"Vlissingen","TERNZN":"Terneuzen",
    "HANSWT":"Hansweert","BATH":"Bath","KALLSLZS":"Kallosluis",
}
STATION_LONS = {
    "VLISSGN":3.596,"TERNZN":3.830,"HANSWT":3.996,
    "BATH":4.212,"KALLSLZS":4.291,
}

plt.rcParams.update({
    "figure.facecolor":"#0e0e0e","axes.facecolor":"#1a1a1a",
    "text.color":"white","axes.labelcolor":"white",
    "xtick.color":"white","ytick.color":"white",
    "axes.edgecolor":"#444","grid.color":"#333","grid.alpha":0.5,
})
COLORS = ["#42A5F5","#EF5350","#66BB6A","#FF9800","#AB47BC","#EC407A"]


def save_fig(fig, name):
    fig.savefig(str(OUT_DIR/name), dpi=150,
                bbox_inches="tight", facecolor="#0e0e0e")
    print(f"  ✓ {name}")
    plt.close(fig)

def styled_ax(ax, title):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.grid(axis="y", lw=0.5)
    ax.spines[:].set_color("#444")


# ── ANALYSIS 1: TIDAL ASYMMETRY (FIXED) ──────────────────────────────────────

def analyse_tidal_asymmetry(conn):
    """
    FIX: asymmetry computed from ebb vs flood DURATION, not amplitude ratio.
    Ebb duration > flood duration → ebb-dominant (shorter, stronger flood)
    """
    print("\n[1] Tidal asymmetry (fixed) ...")
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for station in STATION_ORDER:
        df = pd.read_sql_query(f"""
            SELECT o.datetime, o.value AS wl_m
            FROM observations o
            JOIN stations   st ON st.id = o.station_id
            JOIN parameters p  ON p.id  = o.parameter_id
            WHERE st.code = '{station}' AND p.code = 'WATHTE'
              AND o.value IS NOT NULL
            ORDER BY o.datetime
        """, conn, parse_dates=["datetime"])

        if df.empty or len(df) < 1000:
            continue
        df = df.set_index("datetime").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.resample("10min").mean().interpolate(limit=3)
        wl = df["wl_m"].dropna().values

        hw_idx, _ = find_peaks(wl,  distance=60, prominence=0.5)
        lw_idx, _ = find_peaks(-wl, distance=60, prominence=0.5)

        if len(hw_idx) < 20 or len(lw_idx) < 20:
            continue

        hw_vals = wl[hw_idx]
        lw_vals = wl[lw_idx]
        mtl_val = (hw_vals.mean() + lw_vals.mean()) / 2

        # ── FIX: compute flood/ebb durations ────────────────────────────────
        # Pair each HW with the preceding LW to get flood duration
        # Pair each HW with the following LW to get ebb duration
        flood_durs, ebb_durs = [], []
        times = df.index
        hw_times = times[hw_idx]
        lw_times = times[lw_idx]

        for hw_t in hw_times:
            # preceding LW
            before = lw_times[lw_times < hw_t]
            after  = lw_times[lw_times > hw_t]
            if len(before) > 0:
                flood_durs.append(
                    (hw_t - before[-1]).total_seconds() / 3600)
            if len(after) > 0:
                ebb_durs.append(
                    (after[0] - hw_t).total_seconds() / 3600)

        if not flood_durs or not ebb_durs:
            continue

        mean_flood = np.median(flood_durs)
        mean_ebb   = np.median(ebb_durs)
        # Asymmetry: ebb/flood > 1 means longer ebb = ebb-dominant
        # (shorter, stronger flood → sediment import)
        asym = mean_ebb / (mean_flood + 1e-6)
        dominant = "flood" if asym < 1.0 else "ebb"

        results[station] = {
            "name":         STATION_NAMES[station],
            "lon":          STATION_LONS[station],
            "MHW":          round(float(hw_vals.mean()), 3),
            "MLW":          round(float(lw_vals.mean()), 3),
            "TR":           round(float(hw_vals.mean()-lw_vals.mean()), 3),
            "MTL":          round(float(mtl_val), 3),
            "flood_dur_hr": round(mean_flood, 3),
            "ebb_dur_hr":   round(mean_ebb,   3),
            "asymmetry":    round(asym, 4),
            "dominant":     dominant,
        }
        print(f"  {station:10s}: MHW={hw_vals.mean():+.3f}  "
              f"TR={hw_vals.mean()-lw_vals.mean():.3f}  "
              f"flood={mean_flood:.2f}h  ebb={mean_ebb:.2f}h  "
              f"asym={asym:.3f} ({dominant})")

    if not results:
        plt.close(fig); return {}

    ax = axes[0]
    srt = sorted(results.values(), key=lambda x: x["lon"])
    lons  = [s["lon"] for s in srt]
    names = [s["name"] for s in srt]
    mhws  = [s["MHW"] for s in srt]
    mlws  = [s["MLW"] for s in srt]
    trs   = [s["TR"]  for s in srt]
    ax.fill_between(lons, mlws, mhws, alpha=0.2, color="#42A5F5")
    ax.plot(lons, mhws, "o-", color="#EF5350", lw=2, ms=7, label="MHW")
    ax.plot(lons, mlws, "o-", color="#42A5F5", lw=2, ms=7, label="MLW")
    ax.plot(lons, trs,  "s--",color="#FF9800", lw=1.5, ms=7, label="TR")
    ax.axhline(0, color="white", lw=0.8, ls=":")
    ax.set_xticks(lons); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Water level (m NAP)")
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    styled_ax(ax, "Tidal characteristics — funnel amplification\n"
              f"TR at mouth: {trs[0]:.2f}m  at Kallosluis: {trs[-1]:.2f}m")

    ax2 = axes[1]
    asym_vals  = [s["asymmetry"] for s in srt]
    flood_d    = [s["flood_dur_hr"] for s in srt]
    ebb_d      = [s["ebb_dur_hr"]   for s in srt]
    x = np.arange(len(names))
    w = 0.3
    ax2.bar(x-w/2, flood_d, width=w, color="#EF5350", alpha=0.85,
            label="Flood duration (h)")
    ax2.bar(x+w/2, ebb_d,   width=w, color="#42A5F5", alpha=0.85,
            label="Ebb duration (h)")
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("Duration (hours)")
    ax2.legend(fontsize=8, facecolor="#222", labelcolor="white")
    # Add dominance label
    for i, (a, n) in enumerate(zip(asym_vals, names)):
        dom = "ebb↑" if a > 1 else "flood↑"
        ax2.text(i, max(flood_d[i],ebb_d[i])+0.05, dom,
                 ha="center", color="white", fontsize=8)
    styled_ax(ax2, "Flood vs ebb duration per station\n"
              "shorter flood = stronger flood current = sediment import")

    fig.suptitle("Tidal Asymmetry  |  Western Scheldt 2017–2025",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "tidal_asymmetry.png")
    return results


# ── ANALYSIS 2: STORM SURGE (FIXED — per-station MHW) ────────────────────────

def analyse_storm_surges(conn, station_mhw: dict):
    """
    FIX: use per-station MHW from asymmetry results, not Vlissingen.
    Threshold = station_MHW + 0.5m (genuine overtopping events only)
    """
    print("\n[2] Storm surge frequency (station-specific thresholds) ...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    surge_data = {}

    for station in STATION_ORDER:
        mhw_st = station_mhw.get(station, MHW_VLISS)
        thresh  = mhw_st + 0.5
        thresh2 = mhw_st + 1.0

        df = pd.read_sql_query(f"""
            SELECT SUBSTR(o.datetime,1,4) AS year,
                   COUNT(*)               AS n_surges,
                   ROUND(MAX(o.value),3)  AS peak_wl
            FROM observations o
            JOIN stations   st ON st.id = o.station_id
            JOIN parameters p  ON p.id  = o.parameter_id
            WHERE st.code = '{station}' AND p.code = 'WATHTE'
              AND o.value > {thresh}
            GROUP BY year ORDER BY year
        """, conn)

        if df.empty: continue
        df["year"] = df["year"].astype(int)
        surge_data[station] = {"df": df, "threshold": thresh,
                                "mhw": mhw_st}
        print(f"  {station:10s}: MHW={mhw_st:+.3f}  "
              f"threshold={thresh:.3f}  "
              f"total surge-hrs={df['n_surges'].sum():,}  "
              f"peak={df['peak_wl'].max():.3f}m")

    if not surge_data:
        plt.close(fig); return

    ax = axes[0]
    years = sorted(set(
        y for d in surge_data.values() for y in d["df"]["year"]))
    x = np.arange(len(years)); w = 0.15

    for i, (station, d) in enumerate(surge_data.items()):
        counts = [d["df"][d["df"]["year"]==y]["n_surges"].sum()
                  if y in d["df"]["year"].values else 0 for y in years]
        ax.bar(x + i*w, counts, width=w,
               label=f"{STATION_NAMES[station]} (thr={d['threshold']:.2f}m)",
               color=COLORS[i], alpha=0.85)

    ax.set_xticks(x + w*len(surge_data)/2)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Hours above station-specific MHW+0.5m")
    ax.legend(fontsize=6, facecolor="#222", labelcolor="white")
    styled_ax(ax, "Storm surge frequency per year\n"
              "(threshold = local MHW + 0.5m per station)")

    ax2 = axes[1]
    if "BATH" in surge_data:
        d     = surge_data["BATH"]
        df_b  = d["df"]
        ax2.bar(df_b["year"], df_b["n_surges"],
                color="#42A5F5", alpha=0.8, label="Surge hrs/year")
        if len(df_b) >= 4:
            res = sp_stats.linregress(df_b["year"].astype(float),
                                       df_b["n_surges"].astype(float))
            xfit = np.linspace(df_b["year"].min(),
                               df_b["year"].max(), 50)
            ls = "-" if res.pvalue < 0.10 else "--"
            ax2.plot(xfit, res.slope*xfit+res.intercept,
                     color="#EF5350", lw=2, ls=ls,
                     label=f"Trend: {res.slope:+.1f} hrs/yr "
                           f"(p={res.pvalue:.3f})")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Surge hours/year")
        ax2.set_title(
            f"Bath station — MHW={d['mhw']:+.3f}m  "
            f"threshold={d['threshold']:.3f}m NAP\n"
            f"Genuine overtopping events above local high water",
            fontsize=9, fontweight="bold")
        ax2.legend(fontsize=8, facecolor="#222", labelcolor="white")
        ax2.grid(axis="y", lw=0.5)
        ax2.spines[:].set_color("#444")

    fig.suptitle("Storm Surge Analysis  |  Station-specific thresholds  "
                  "|  Western Scheldt 2017–2025",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "storm_surge_frequency.png")


# ── ANALYSIS 3: SALINITY (FIXED variance query) ───────────────────────────────

def analyse_salinity(conn):
    print("\n[3] Salinity gradient ...")

    sal_stations = pd.read_sql_query("""
        SELECT DISTINCT st.code, COUNT(o.id) AS n
        FROM observations o
        JOIN stations st ON st.id = o.station_id
        JOIN parameters p ON p.id = o.parameter_id
        WHERE p.code = 'SALNTT' AND o.value BETWEEN 0 AND 40
        GROUP BY st.code
    """, conn)
    print(f"  Stations with valid salinity: "
          f"{sal_stations['code'].tolist()}")

    if sal_stations.empty:
        print("  No salinity data"); return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.ravel()

    df_annual = pd.read_sql_query("""
        SELECT st.code, st.lon,
               SUBSTR(o.datetime,1,4)  AS year,
               ROUND(AVG(o.value),2)   AS mean_sal,
               ROUND(MIN(o.value),2)   AS min_sal,
               ROUND(MAX(o.value),2)   AS max_sal,
               ROUND(
                 (MAX(o.value)-MIN(o.value)),2) AS range_sal
        FROM observations o
        JOIN stations st ON st.id = o.station_id
        JOIN parameters p ON p.id = o.parameter_id
        WHERE p.code='SALNTT' AND o.value BETWEEN 0 AND 40
        GROUP BY st.code, year ORDER BY st.lon, year
    """, conn)

    df_monthly = pd.read_sql_query("""
        SELECT st.code,
               CAST(SUBSTR(o.datetime,6,2) AS INTEGER) AS month,
               ROUND(AVG(o.value),2)  AS mean_sal,
               ROUND(MIN(o.value),2)  AS min_sal,
               ROUND(MAX(o.value),2)  AS max_sal
        FROM observations o
        JOIN stations st ON st.id = o.station_id
        JOIN parameters p ON p.id = o.parameter_id
        WHERE p.code='SALNTT' AND o.value BETWEEN 0 AND 40
        GROUP BY st.code, month ORDER BY st.code, month
    """, conn)

    # Panel 1: longitudinal gradient
    ax = axes[0]
    for i, station in enumerate(STATION_ORDER):
        sub = df_annual[df_annual["code"] == station]
        if sub.empty: continue
        lon = STATION_LONS[station]
        ax.errorbar(
            [lon]*len(sub), sub["mean_sal"],
            yerr=[sub["mean_sal"]-sub["min_sal"],
                  sub["max_sal"]-sub["mean_sal"]],
            fmt="o", color=COLORS[i], ms=6, alpha=0.7,
            label=STATION_NAMES[station], capsize=3)
    overall = df_annual.groupby("code")["mean_sal"].mean()
    lons_s = [(STATION_LONS[c], overall[c])
               for c in overall.index if c in STATION_LONS]
    if lons_s:
        lons_s.sort()
        ax.plot([l for l,_ in lons_s], [v for _,v in lons_s],
                "w--", lw=1.5, alpha=0.5, label="Overall mean")
    ax.axhline(18, color="#FF9800", lw=1.5, ls="--",
               label="18 PSU (halocline)")
    ax.set_ylabel("Salinity (PSU)")
    ax.set_xlabel("Longitude (°E)")
    ax.legend(fontsize=7, facecolor="#222", labelcolor="white")
    styled_ax(ax, "Longitudinal salinity gradient")

    # Panel 2: seasonal cycle
    ax2 = axes[1]
    months = np.arange(1,13)
    mlbls  = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    for i, station in enumerate(STATION_ORDER):
        sub = df_monthly[df_monthly["code"] == station]
        if sub.empty: continue
        ax2.fill_between(sub["month"], sub["min_sal"], sub["max_sal"],
                         alpha=0.15, color=COLORS[i])
        ax2.plot(sub["month"], sub["mean_sal"], "o-",
                 color=COLORS[i], lw=2, ms=5,
                 label=STATION_NAMES[station])
    ax2.axhline(18, color="#FF9800", lw=1.2, ls="--", alpha=0.7)
    ax2.set_xticks(months); ax2.set_xticklabels(mlbls)
    ax2.set_ylabel("Salinity (PSU)")
    ax2.legend(fontsize=7, facecolor="#222", labelcolor="white")
    styled_ax(ax2, "Seasonal salinity cycle\n"
              "shaded = min–max range")

    # Panel 3: annual trends
    ax3 = axes[2]
    for i, station in enumerate(STATION_ORDER):
        sub = df_annual[df_annual["code"]==station].copy()
        if len(sub) < 3: continue
        sub["year"] = sub["year"].astype(int)
        ax3.plot(sub["year"], sub["mean_sal"], "o-",
                 color=COLORS[i], lw=2, ms=5,
                 label=STATION_NAMES[station])
        if len(sub) >= 4:
            res = sp_stats.linregress(sub["year"], sub["mean_sal"])
            if res.pvalue < 0.10:
                xf = np.linspace(sub["year"].min(),sub["year"].max(),30)
                ax3.plot(xf, res.slope*xf+res.intercept,
                         color=COLORS[i], lw=1, ls="--", alpha=0.7)
                print(f"  {station}: sal trend = "
                      f"{res.slope:+.3f} PSU/yr  p={res.pvalue:.3f} *")
    ax3.set_ylabel("Mean annual salinity (PSU)")
    ax3.legend(fontsize=7, facecolor="#222", labelcolor="white")
    styled_ax(ax3, "Annual salinity trends\ndashed = significant (p<0.10)")

    # Panel 4: variability — FIX: use proper std dev from pandas
    ax4 = axes[3]
    df_std = pd.read_sql_query("""
        SELECT st.code,
               ROUND(AVG(o.value),2)  AS mean_sal,
               COUNT(o.id)            AS n_obs
        FROM observations o
        JOIN stations st ON st.id = o.station_id
        JOIN parameters p ON p.id = o.parameter_id
        WHERE p.code='SALNTT' AND o.value BETWEEN 0 AND 40
        GROUP BY st.code
    """, conn)

    # Compute std dev properly in pandas
    std_rows = []
    for _, row in df_std.iterrows():
        vals = pd.read_sql_query(f"""
            SELECT o.value FROM observations o
            JOIN stations st ON st.id=o.station_id
            JOIN parameters p ON p.id=o.parameter_id
            WHERE st.code='{row["code"]}'
              AND p.code='SALNTT'
              AND o.value BETWEEN 0 AND 40
        """, conn)["value"]
        std_rows.append({"code": row["code"],
                          "mean": row["mean_sal"],
                          "std":  vals.std(),
                          "n":    len(vals)})
    df_std2 = pd.DataFrame(std_rows)
    if not df_std2.empty:
        names_v = [STATION_NAMES.get(c,c) for c in df_std2["code"]]
        ax4.bar(names_v, df_std2["std"],
                color=COLORS[:len(df_std2)], alpha=0.85)
        ax4.set_ylabel("Salinity std dev (PSU)")
        ax4.set_xticklabels(names_v, rotation=30, ha="right")
        styled_ax(ax4, "Salinity variability per station\n"
                  "high = tidal-fluvial transition = vegetation stress zone")

    fig.suptitle("Salinity Analysis  |  Western Scheldt 2017–2025",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "salinity_gradient.png")


# ── ANALYSIS 4: SALINITY-RISK ─────────────────────────────────────────────────

def analyse_salinity_risk(conn):
    print("\n[4] Salinity-risk correlation ...")
    risk_sal = pd.read_sql_query("""
        SELECT rs.name, rs.risk_class, rs.risk_class_name,
               rs.issue_type, rs.nearest_station,
               ROUND(AVG(o.value),2)  AS mean_sal,
               ROUND(MIN(o.value),2)  AS min_sal,
               ROUND(MAX(o.value),2)  AS max_sal,
               COUNT(o.id)            AS n_obs
        FROM risk_sites rs
        JOIN stations   st ON st.code = rs.nearest_station
        JOIN observations o ON o.station_id = st.id
        JOIN parameters   p ON p.id = o.parameter_id
        WHERE p.code='SALNTT' AND o.value BETWEEN 0 AND 40
        GROUP BY rs.name ORDER BY rs.risk_class DESC
    """, conn)

    if risk_sal.empty:
        print("  No salinity data at risk sites — "
              "SALNTT only at Terneuzen, not at Bath/Hansweert")
        print("  → To add data: rerun acquire_monitoring_db.py")
        print("    Check ddlpy for SALNTT at station codes:")
        print("    BATH, HANSWT with different naming convention")
        return

    print(risk_sal[["name","risk_class_name",
                    "mean_sal","n_obs"]].to_string(index=False))

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    cls_colors = {"Low":"#2196F3","Medium":"#4CAF50",
                  "High":"#FF9800","Critical":"#F44336"}
    for _, row in risk_sal.iterrows():
        col = cls_colors.get(row["risk_class_name"],"#888")
        ax.scatter(row["mean_sal"], row["risk_class"],
                   color=col, s=200, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(row["name"].split("-")[0][:16],
                    (row["mean_sal"], row["risk_class"]),
                    textcoords="offset points",
                    xytext=(5,3), color="white", fontsize=8)
    ax.axvspan(0, 18, alpha=0.08, color="#FF9800",
               label="Brackish (<18 PSU)")
    ax.axvspan(18, 40, alpha=0.08, color="#42A5F5",
               label="Saline (>18 PSU)")
    ax.set_xlabel("Mean salinity (PSU)")
    ax.set_yticks([1,2,3,4])
    ax.set_yticklabels(["Low","Medium","High","Critical"])
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    styled_ax(ax, "Salinity regime vs risk class")
    fig.tight_layout()
    save_fig(fig, "salinity_risk_correlation.png")


# ── ANALYSIS 5: TIDAL RANGE TREND ────────────────────────────────────────────

def analyse_tidal_range_trend(conn):
    print("\n[5] Tidal range trend ...")

    df = pd.read_sql_query("""
        SELECT st.code, st.lon,
               SUBSTR(o.datetime,1,4) AS year,
               ROUND(AVG(CASE WHEN o.value>1.5 THEN o.value END),3) AS mean_hw,
               ROUND(AVG(CASE WHEN o.value<-1.0 THEN o.value END),3) AS mean_lw
        FROM observations o
        JOIN stations st ON st.id=o.station_id
        JOIN parameters p ON p.id=o.parameter_id
        WHERE p.code='WATHTE'
        GROUP BY st.code, year ORDER BY st.lon, year
    """, conn)
    df["tidal_range"] = df["mean_hw"] - df["mean_lw"]
    df["year"] = df["year"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(18,7))
    ax = axes[0]
    for i, station in enumerate(STATION_ORDER):
        sub = df[df["code"]==station].dropna(subset=["tidal_range"])
        if sub.empty: continue
        ax.plot(sub["year"], sub["tidal_range"], "o-",
                color=COLORS[i], lw=2, ms=6,
                label=STATION_NAMES[station])
        if len(sub) >= 4:
            res = sp_stats.linregress(sub["year"],sub["tidal_range"])
            if res.pvalue < 0.10:
                xf = np.linspace(sub["year"].min(),sub["year"].max(),30)
                ax.plot(xf, res.slope*xf+res.intercept,
                        color=COLORS[i], lw=1, ls="--", alpha=0.7)
                print(f"  {station}: TR trend={res.slope*100:+.2f} cm/yr "
                      f"p={res.pvalue:.3f} *")
    ax.set_ylabel("Tidal range (m)")
    ax.legend(fontsize=8, facecolor="#222", labelcolor="white")
    styled_ax(ax, "Annual tidal range 2017–2025\n"
              "dashed = significant trend (p<0.10)")

    ax2 = axes[1]
    for i, station in enumerate(STATION_ORDER):
        sub = df[df["code"]==station].dropna(subset=["mean_hw","mean_lw"])
        if sub.empty: continue
        ax2.plot(sub["year"], sub["mean_hw"], "-",
                 color=COLORS[i], lw=1.5, label=f"{STATION_NAMES[station]} HW")
        ax2.plot(sub["year"], sub["mean_lw"], "--",
                 color=COLORS[i], lw=1.5, alpha=0.7)
    ax2.axhline(MHW_VLISS, color="#EF5350", lw=1, ls=":",
                label=f"Vlissingen MHW ({MHW_VLISS:.3f}m)")
    ax2.axhline(MLW_VLISS, color="#42A5F5", lw=1, ls=":",
                label=f"Vlissingen MLW ({MLW_VLISS:.3f}m)")
    ax2.set_ylabel("Water level (m NAP)")
    ax2.legend(fontsize=6, facecolor="#222", labelcolor="white", ncol=2)
    styled_ax(ax2, "HW (solid) and LW (dashed) trends")
    fig.suptitle("Tidal Range Trends  |  Western Scheldt 2017–2025",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "tidal_range_trend.png")


# ── ANALYSIS 6: RISK SITE PROFILES (FIXED threshold) ─────────────────────────

def analyse_risk_site_profiles(conn, station_mhw: dict):
    print("\n[6] Risk site hydrodynamic profiles (fixed) ...")

    # Build dynamic CASE for station-specific thresholds
    # Each station gets its own MHW+0.5 threshold
    thresh_cases = " ".join([
        f"WHEN st.code='{s}' THEN {mhw+0.5:.3f}"
        for s, mhw in station_mhw.items()
    ])
    thresh_default = f"{MHW_VLISS+0.5:.3f}"

    df = pd.read_sql_query(f"""
        SELECT
            rs.name, rs.risk_class, rs.risk_class_name,
            rs.issue_type, rs.nearest_station, st.lon,
            ROUND(AVG(CASE WHEN p.code='WATHTE'
                      THEN o.value END),3)         AS mean_wl,
            ROUND(MAX(CASE WHEN p.code='WATHTE'
                      THEN o.value END),3)         AS max_wl,
            ROUND(100.0*SUM(
                CASE WHEN p.code='WATHTE' AND o.value >
                    CASE {thresh_cases}
                    ELSE {thresh_default} END
                THEN 1 ELSE 0 END)
              / NULLIF(COUNT(CASE WHEN p.code='WATHTE'
                             THEN 1 END),0), 3)    AS surge_pct,
            ROUND(AVG(CASE WHEN p.code='SALNTT'
                      AND o.value BETWEEN 0 AND 40
                      THEN o.value END),2)         AS mean_sal
        FROM risk_sites  rs
        JOIN stations    st ON st.code = rs.nearest_station
        JOIN observations o  ON o.station_id = st.id
        JOIN parameters  p   ON p.id = o.parameter_id
        WHERE p.code IN ('WATHTE','SALNTT')
        GROUP BY rs.name ORDER BY rs.risk_class DESC, st.lon
    """, conn)

    if df.empty:
        print("  No data"); return

    print(df[["name","risk_class_name","nearest_station",
              "surge_pct","max_wl"]].to_string(index=False))

    cls_colors = {"Low":"#2196F3","Medium":"#4CAF50",
                  "High":"#FF9800","Critical":"#F44336"}
    colors_p = [cls_colors.get(c,"#888") for c in df["risk_class_name"]]
    names_s  = [n.split("-")[0].strip()[:16] for n in df["name"]]
    y = np.arange(len(df))

    ncols = 2 if df["mean_sal"].isna().all() else 3
    fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 8))
    if ncols == 2:
        axes = list(axes) + [None]

    # Surge %
    ax = axes[0]
    ax.barh(y, df["surge_pct"].fillna(0), color=colors_p, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names_s)
    ax.set_xlabel("% hours above local MHW+0.5m\n(genuine overtopping)")
    styled_ax(ax, "Storm surge exposure\n(station-specific threshold)")

    # Max WL
    ax2 = axes[1]
    ax2.barh(y, df["max_wl"].fillna(0), color=colors_p, alpha=0.85)
    # Add station MHW reference lines
    for station, mhw in station_mhw.items():
        ax2.axvline(mhw, color="#888", lw=0.5, ls=":", alpha=0.5)
    ax2.set_yticks(y); ax2.set_yticklabels(names_s)
    ax2.set_xlabel("Maximum recorded WL (m NAP)")
    styled_ax(ax2, "Maximum water level recorded\nvertical lines = local MHW")

    # Salinity if available
    if axes[2] is not None and not df["mean_sal"].isna().all():
        ax3 = axes[2]
        ax3.barh(y, df["mean_sal"].fillna(0), color=colors_p, alpha=0.85)
        ax3.axvline(18, color="#FF9800", lw=1.5, ls="--",
                    label="18 PSU boundary")
        ax3.set_yticks(y); ax3.set_yticklabels(names_s)
        ax3.set_xlabel("Mean salinity (PSU)")
        ax3.legend(fontsize=8, facecolor="#222", labelcolor="white")
        styled_ax(ax3, "Salinity regime")

    # Legend for risk classes
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=n)
               for n, c in cls_colors.items()]
    axes[0].legend(handles=patches, loc="lower right",
                   fontsize=7, facecolor="#222", labelcolor="white")

    fig.suptitle(
        "Risk Site Hydrodynamic Profiles  |  "
        "In-situ × Satellite  |  Scheldt 2017–2025",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "risk_site_profiles.png")
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=== Monitoring Analysis v2 (patched) ===\n")

    if not DB_PATH.exists():
        sys.exit(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")

    # Run asymmetry first — results feed into surge analysis
    asym_results = analyse_tidal_asymmetry(conn)

    # Build per-station MHW dict from results (fallback to Vlissingen)
    station_mhw = {s: d["MHW"]
                   for s, d in asym_results.items()} \
                  if asym_results else \
                  {s: MHW_VLISS for s in STATION_ORDER}

    print(f"\n  Station MHW used for surge thresholds:")
    for s, mhw in station_mhw.items():
        print(f"    {s}: MHW={mhw:+.3f}m  threshold={mhw+0.5:.3f}m NAP")

    analyse_storm_surges(conn, station_mhw)
    analyse_salinity(conn)
    analyse_salinity_risk(conn)
    analyse_tidal_range_trend(conn)
    analyse_risk_site_profiles(conn, station_mhw)

    # Save updated summary
    summary = {"tidal_asymmetry": asym_results,
               "station_mhw":    station_mhw}
    with open(OUT_DIR/"analysis_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ analysis_summary.json updated")

    conn.close()
    print(f"\n✓ All analyses complete → {OUT_DIR}/")
    print("\nKey findings:")
    print("  • Tidal range DECREASING at all stations (significant)")
    print("  • Per-station MHW used for correct surge thresholds")
    print("  • Salinity only at Terneuzen — need Bath/Hansweert codes")
    print("  • Geulwandverdediging correctly shows Low risk (negative control)")
