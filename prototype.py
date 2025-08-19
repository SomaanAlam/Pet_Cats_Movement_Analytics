#!/usr/bin/env python3
# Pet Cats Movement Analytics — Prototype
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import folium
from folium.plugins import MarkerCluster

# ------------------------------
# ESA WorldCover classes (subset)
# ------------------------------
ESA_WORLDCOVER = {
    0: "No data",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse veg.",
    70: "Snow & ice",
    80: "Water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss & lichen",
}

# ------------------------------
# Utility functions
# ------------------------------
def normalize_headers(df: pd.DataFrame):
    """Create lookup from normalized -> original header."""
    norm = {}
    for c in df.columns:
        key = " ".join(c.replace("\xa0", " ").strip().lower().split())
        norm[key] = c
    return norm

def pick_column(norm_map, candidates):
    for cand in candidates:
        if not cand:
            continue
        key = " ".join(str(cand).strip().lower().split())
        if key in norm_map:
            return norm_map[key]
    return None

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(phi2)
    x = np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dl)
    brng = np.degrees(np.arctan2(y, x))
    return (brng + 360) % 360

def turning_angle_deg(b1, b2):
    return ((b2 - b1 + 180) % 360) - 180

# ------------------------------
# Core steps
# ------------------------------
def auto_speed_filter(df, max_speed_mps=5.0):
    df = df.sort_values(["id", "timestamp"]).copy()
    df["lat_prev"] = df.groupby("id")["latitude"].shift(1)
    df["lon_prev"] = df.groupby("id")["longitude"].shift(1)
    df["t_prev"]   = df.groupby("id")["timestamp"].shift(1)
    df["dist_m"] = haversine_m(df["latitude"], df["longitude"], df["lat_prev"], df["lon_prev"])
    df["dt_s"] = (df["timestamp"] - df["t_prev"]).dt.total_seconds()
    df["speed_mps"] = df["dist_m"] / df["dt_s"]
    keep = df["t_prev"].isna() | (df["speed_mps"].fillna(0) <= max_speed_mps)
    return df.loc[keep, ["id", "timestamp", "latitude", "longitude"]].copy()

def resample_interpolate(df, interval_s=300):
    out = []
    for gid, g in df.groupby("id"):
        g = g.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        g = g.set_index("timestamp")
        if g.index.tz is None:
            g.index = g.index.tz_localize("UTC")
        idx = pd.date_range(g.index.min(), g.index.max(), freq=f"{interval_s}s", tz="UTC")
        g2 = g.reindex(idx)
        g2["latitude"]  = g2["latitude"].interpolate()
        g2["longitude"] = g2["longitude"].interpolate()
        g2["id"] = gid
        g2 = g2.reset_index().rename(columns={"index": "timestamp"})
        out.append(g2[["id", "timestamp", "latitude", "longitude"]])
    return pd.concat(out, ignore_index=True)

def derive_movement(df):
    df = df.sort_values(["id", "timestamp"]).copy()
    df["lat_prev"] = df.groupby("id")["latitude"].shift(1)
    df["lon_prev"] = df.groupby("id")["longitude"].shift(1)
    df["t_prev"]   = df.groupby("id")["timestamp"].shift(1)
    df["dist_m"]   = haversine_m(df["latitude"], df["longitude"], df["lat_prev"], df["lon_prev"])
    df["dt_s"]     = (df["timestamp"] - df["t_prev"]).dt.total_seconds()
    df["bearing"]  = bearing_deg(df["lat_prev"], df["lon_prev"], df["latitude"], df["longitude"])
    df["bearing_prev"] = df.groupby("id")["bearing"].shift(1)
    df["turn_deg"] = turning_angle_deg(df["bearing_prev"], df["bearing"])
    df["speed_mps"] = df["dist_m"] / df["dt_s"]
    return df

def detect_dwell(df, radius_m=25, min_points=3):
    df = df.sort_values(["id", "timestamp"]).copy()
    df["dwell"] = False
    half = min_points // 2
    for gid, g in df.groupby("id"):
        coords = g[["latitude", "longitude"]].to_numpy()
        flags = np.zeros(len(g), dtype=bool)
        for i in range(len(g)):
            s = max(0, i - half)
            e = min(len(g), i + half + 1)
            center = coords[i]
            d = haversine_m(coords[s:e, 0], coords[s:e, 1], center[0], center[1])
            if np.nanmax(d) <= radius_m and (e - s) >= min_points:
                flags[i] = True
        df.loc[g.index, "dwell"] = flags
    return df

def sample_rasters(df, dem_path=None, lc_path=None):
    if rasterio is None or (dem_path is None and lc_path is None):
        return df.copy()
    df = df.copy()
    coords = list(zip(df["longitude"].tolist(), df["latitude"].tolist()))
    if dem_path and os.path.exists(dem_path):
        with rasterio.open(dem_path) as src:
            nodata = src.nodata
            vals = [v[0] if v is not None else np.nan for v in src.sample(coords)]
            # mask common nodata explicitly too
            df["elevation"] = [
                (np.nan if (x is None or (nodata is not None and x == nodata) or x in (-10000, -9999)) else float(x))
                for x in vals
            ]
    if lc_path and os.path.exists(lc_path):
        with rasterio.open(lc_path) as src:
            vals = [v[0] if v is not None else np.nan for v in src.sample(coords)]
            df["landcover"] = vals
            # best effort: int codes when possible
            try:
                df["landcover"] = df["landcover"].astype("Int64")
            except Exception:
                pass
            df["landcover_class"] = df["landcover"].map(ESA_WORLDCOVER)
    return df

# ------------------------------
# Plotting
# ------------------------------
def plot_hist(series, title, path, log_x=False, xlabel=None):
    s = series.dropna()
    if s.empty:
        return
    plt.figure()
    if sns is not None:
        sns.histplot(s, bins=40, kde=True)
    else:
        s.plot(kind="hist", bins=40)
    if log_x and (s > 0).any():
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel or series.name)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_landcover_bar(df, path):
    if "landcover_class" not in df.columns:
        return
    s = df["landcover_class"].dropna()
    if s.empty:
        return
    counts = s.value_counts().sort_values(ascending=False)
    plt.figure()
    if sns is not None:
        sns.barplot(x=counts.values, y=counts.index)
        plt.xlabel("fix count")
        plt.ylabel("land-cover")
    else:
        plt.barh(range(len(counts)), counts.values)
        plt.yticks(range(len(counts)), counts.index)
        plt.xlabel("fix count")
        plt.ylabel("land-cover")
    plt.title("Land cover along tracks (ESA WorldCover)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_tracks(df, title, path):
    """Static PNG without a huge legend."""
    if gpd is None:
        plt.figure()
        if sns is not None and "id" in df.columns:
            sns.scatterplot(data=df, x="longitude", y="latitude", hue="id", s=6, linewidth=0, legend=False)
        else:
            plt.scatter(df["longitude"], df["latitude"], s=4)
        plt.title(title)
        plt.xlabel("longitude"); plt.ylabel("latitude")
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        return

    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    g3857 = g.to_crs(3857)
    plt.figure(figsize=(7,7))
    ax = plt.gca()
    # IMPORTANT: legend=False to avoid the massive legend
    if "id" in g3857.columns:
        g3857.plot(ax=ax, column="id", markersize=6, alpha=0.8, legend=False, categorical=True)
    else:
        g3857.plot(ax=ax, markersize=6, alpha=0.8)
    if ctx is not None:
        try:
            ctx.add_basemap(ax, crs=g3857.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception:
            pass
    ax.set_axis_off()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_tracks_interactive(df, interval, out_html):
    """Create an interactive HTML map with hover/click popups for id + timestamp."""

    # Center on mean position
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)

    # Cluster markers for performance
    cluster = MarkerCluster(name=f"Tracks {interval}s").add_to(m) if MarkerCluster else m

    # Add points with popups
    for _, row in df.iterrows():
        popup = folium.Popup(
            f"<b>ID:</b> {row.get('id','')}<br>"
            f"<b>Time:</b> {row.get('timestamp','')}", max_width=300
        )
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=3,
            color="#2A81CB",
            fill=True,
            fill_opacity=0.6,
            popup=popup
        ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)
    m.save(out_html)

# ------------------------------
# Home-range proxies
# ------------------------------
def mcp_area_ha(df):
    if gpd is None:
        return pd.DataFrame({"id": df["id"].unique(), "mcp_area_ha": np.nan})
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326").to_crs(3857)
    out = []
    for gid, gg in g.groupby("id"):
        hull = gg.unary_union.convex_hull
        out.append({"id": gid, "mcp_area_ha": hull.area / 10000.0})
    return pd.DataFrame(out)

def kde_proxy_area_ha(df, bandwidth_m=100):
    if gpd is None:
        return pd.DataFrame({"id": df["id"].unique(), "kde_proxy_area_ha": np.nan})
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326").to_crs(3857)
    out = []
    for gid, gg in g.groupby("id"):
        union = gg.buffer(bandwidth_m).unary_union
        out.append({"id": gid, "kde_proxy_area_ha": union.area / 10000.0})
    return pd.DataFrame(out)

# ------------------------------
# Reporting helper
# ------------------------------
def to_md_or_string(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="Path to Movebank-style CSV")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--resample", nargs="+", type=int, default=[300], help="One or more resample intervals in seconds, e.g. 60 300 600")
    ap.add_argument("--id-col", default=None, help="Override ID column")
    ap.add_argument("--time-col", default=None, help="Override timestamp column")
    ap.add_argument("--lat-col", default=None, help="Override latitude column")
    ap.add_argument("--lon-col", default=None, help="Override longitude column")
    ap.add_argument("--dem", default=None, help="DEM GeoTIFF (optional)")
    ap.add_argument("--landcover", default=None, help="ESA WorldCover GeoTIFF (optional)")
    ap.add_argument("--max-speed-mps", type=float, default=5.0, help="Speed spike filter threshold")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.tracks)

    # Auto-detect columns
    norm = normalize_headers(df)
    cand_id   = [args.id_col]   if args.id_col   else ["individual-local-identifier","animal id","tag-local-identifier","individual-id","id"]
    cand_time = [args.time_col] if args.time_col else ["timestamp","event-time"]
    cand_lat  = [args.lat_col]  if args.lat_col  else ["location-lat","location_lat","latitude","lat"]
    cand_lon  = [args.lon_col]  if args.lon_col  else ["location-long","location_long","longitude","lon"]
    col_id   = pick_column(norm, cand_id)
    col_time = pick_column(norm, cand_time)
    col_lat  = pick_column(norm, cand_lat)
    col_lon  = pick_column(norm, cand_lon)
    missing = [name for name,val in [("ID",col_id),("time",col_time),("latitude",col_lat),("longitude",col_lon)] if val is None]
    if missing:
        print("ERROR: Could not auto-detect required columns:", ", ".join(missing), file=sys.stderr)
        print("\nColumns in CSV:", file=sys.stderr)
        for c in df.columns: print(" -", c, file=sys.stderr)
        print('\nHint: pass flags like --id-col "individual local identifier" --time-col "timestamp" --lat-col "location lat" --lon-col "location long"', file=sys.stderr)
        sys.exit(2)

    # Standardize columns & parse times
    df = df.rename(columns={col_id:"id", col_time:"timestamp", col_lat:"latitude", col_lon:"longitude"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","latitude","longitude"]).sort_values(["id","timestamp"])
    df = df.drop_duplicates(subset=["id","timestamp"])

    # Clean spikes
    df_clean = auto_speed_filter(df, max_speed_mps=args.max_speed_mps)

    # Process each resample interval
    for interval in args.resample:
        # Resample & metrics
        df_rs = resample_interpolate(df_clean, interval_s=interval)
        df_mv = derive_movement(df_rs)
        df_dw = detect_dwell(df_mv, radius_m=25, min_points=3)

        # Raster enrichment
        df_enriched = sample_rasters(df_dw, dem_path=args.dem, lc_path=args.landcover)

        # Home-range proxies
        mcp = mcp_area_ha(df_enriched)
        kde = kde_proxy_area_ha(df_enriched, bandwidth_m=100)

        # Save enriched points & summaries
        df_enriched.to_csv(os.path.join(args.out, f"tracks_enriched_{interval}s.csv"), index=False)
        mcp.to_csv(os.path.join(args.out, f"home_range_mcp_{interval}s.csv"), index=False)
        kde.to_csv(os.path.join(args.out, f"home_range_kde_proxy_{interval}s.csv"), index=False)

        summary = df_enriched.groupby("id").agg(
            n_points=("id","size"),
            mean_step_m=("dist_m","mean"),
            median_step_m=("dist_m","median"),
            mean_speed_mps=("speed_mps","mean"),
            dwell_points=("dwell", lambda x: int(x.sum()))
        ).reset_index()
        summary.to_csv(os.path.join(args.out, f"summary_{interval}s.csv"), index=False)

        # Plots
        plot_tracks(df_enriched, f"Tracks (interval={interval}s)", os.path.join(args.out, f"tracks_scatter_{interval}s.png"))
        plot_hist(df_enriched["dist_m"], f"Step length (m) — interval {interval}s", os.path.join(args.out, f"hist_step_length_{interval}s.png"),log_x=True, xlabel="step length (m)")
        plot_hist(df_enriched["speed_mps"], f"Speed (m/s) — interval {interval}s", os.path.join(args.out, f"hist_speed_{interval}s.png"),log_x=True, xlabel="speed (m/s)")
        if "elevation" in df_enriched.columns:
            plot_hist(df_enriched["elevation"], f"Elevation (m) — interval {interval}s", os.path.join(args.out, f"hist_elevation_{interval}s.png"), log_x=False, xlabel="elevation (m)")
        if "landcover_class" in df_enriched.columns:
            plot_landcover_bar(df_enriched, os.path.join(args.out, f"landcover_bar_{interval}s.png"))
        plot_tracks_interactive(df_enriched, interval=interval, out_html=os.path.join(args.out, f"tracks_scatter_{interval}s.html"))

        # Report
        report_lines = []
        report_lines.append(f"# Movement Summary (interval = {interval}s)")
        report_lines.append("")
        report_lines.append("**Key indicators**")
        report_lines.append(to_md_or_string(summary))
        report_lines.append("")
        if not mcp.empty:
            report_lines.append("**Home-range proxies**")
            report_lines.append(to_md_or_string(pd.merge(mcp, kde, on="id", how="outer")))
        report_lines.append("")
        report_lines.append("**Notes**")
        report_lines.append(f"- Speed filter at {args.max_speed_mps} m/s removed spikes.")
        if args.dem and os.path.exists(args.dem):
            report_lines.append("- Elevation from DEM; nodata masked.")
        if args.landcover and os.path.exists(args.landcover):
            report_lines.append("- Land cover from ESA WorldCover (see bar chart).")
        with open(os.path.join(args.out, f"report_{interval}s.md"), "w") as f:
            f.write("\n".join(report_lines))

        print(f"Done for interval {interval}s → outputs in {args.out}")

if __name__ == "__main__":
    main()
