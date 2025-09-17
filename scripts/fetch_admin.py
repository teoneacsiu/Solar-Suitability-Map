# scripts/fetch_admin.py
"""
Create:
  - data/admin/ro_counties.geojson  (Romania județe, Natural Earth Admin-1)
  - data/admin/ro_cities.geojson    (top-N cities by population from OSM)
Usage:
  python -m scripts.fetch_admin --topn 20
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "admin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NE_ADMIN1_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_admin_1_states_provinces.geojson"
)
OVERPASS = "https://overpass-api.de/api/interpreter"

def save_geojson(path: Path, features: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    with path.open("w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)
    print(f"✔ wrote {path}  ({len(features)} features)")

# -------------------- counties (Natural Earth) --------------------
def build_counties() -> list[dict]:
    print("[*] Downloading Natural Earth Admin-1…")
    r = requests.get(NE_ADMIN1_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    feats = []
    for f in data.get("features", []):
        props = f.get("properties", {})
        if props.get("admin") != "Romania":
            continue
        name = (
            props.get("name")
            or props.get("name_en")
            or props.get("name_local")
            or props.get("gn_name")
        )
        # normalize props (keep it tiny)
        new_props = {
            "name": name,
            "iso_3166_2": props.get("iso_3166_2"),
            "type": props.get("type"),  # e.g., county or municipality
        }
        feats.append({"type": "Feature", "geometry": f["geometry"], "properties": new_props})

    # Bucharest is present in NE as a first-order unit; ensure it’s included
    if not any("Bucharest" in (ft["properties"]["name"] or "") for ft in feats):
        print("[WARN] Bucharest not found in NE file — but this is unusual.")

    return feats

# -------------------- top cities (OSM / Overpass) --------------------
def overpass_top_cities(n: int) -> list[dict]:
    print("[*] Querying Overpass for cities/towns with population…")
    q = """
    [out:json][timeout:90];
    area["ISO3166-1"="RO"]->.a;
    (
      node["place"="city"]["population"](area.a);
      node["place"="town"]["population"](area.a);
    );
    out tags center;
    """
    r = requests.post(OVERPASS, data=q.encode("utf-8"), timeout=120)
    r.raise_for_status()
    data = r.json()
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name")
        pop = tags.get("population")
        if not name or not pop:
            continue
        # convert population safely
        try:
            pop_i = int(str(pop).replace(" ", "").replace(",", ""))
        except Exception:
            continue
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        rows.append((pop_i, name, float(lat), float(lon)))

    if not rows:
        raise RuntimeError("Overpass returned no city points with population.")

    rows.sort(reverse=True)  # by population
    rows = rows[:n]

    feats = []
    for pop_i, name, lat, lon in rows:
        props = {"name": name, "population": pop_i}
        geom = {"type": "Point", "coordinates": [lon, lat]}
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    return feats

# Fallback list if Overpass is down (approx coords & pops)
FALLBACK_20 = [
    ("București", 1883000, 44.4268, 26.1025),
    ("Cluj-Napoca", 286598, 46.7712, 23.6236),
    ("Timișoara", 250849, 45.7489, 21.2087),
    ("Iași", 271692, 47.1585, 27.6014),
    ("Constanța", 283872, 44.1598, 28.6348),
    ("Craiova", 234221, 44.3302, 23.7949),
    ("Brașov", 237589, 45.6579, 25.6012),
    ("Galați", 249432, 45.4353, 28.0079),
    ("Ploiești", 209945, 44.9416, 26.0231),
    ("Oradea", 196367, 47.0722, 21.9217),
    ("Brăila", 180302, 45.2692, 27.9575),
    ("Arad", 159074, 46.1866, 21.3123),
    ("Pitești", 155383, 44.8565, 24.8692),
    ("Sibiu", 147245, 45.7983, 24.1256),
    ("Bacău", 144307, 46.5670, 26.9138),
    ("Târgu Mureș", 134290, 46.5425, 24.5575),
    ("Baia Mare", 123738, 47.6597, 23.5795),
    ("Buzău", 115494, 45.1500, 26.8167),
    ("Botoșani", 106847, 47.7484, 26.6694),
    ("Satu Mare", 102411, 47.7928, 22.8857),
]

def fallback_cities(n: int) -> list[dict]:
    rows = FALLBACK_20[:n]
    feats = []
    for name, pop, lat, lon in rows:
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "population": pop},
        })
    return feats

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", type=int, default=20, help="How many cities to keep")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true", help="Overwrite if exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counties_path = out_dir / "ro_counties.geojson"
    cities_path   = out_dir / "ro_cities.geojson"

    # Counties
    if counties_path.exists() and not args.force:
        print(f"[skip] {counties_path} exists")
    else:
        feats = build_counties()
        save_geojson(counties_path, feats)

    # Cities
    if cities_path.exists() and not args.force:
        print(f"[skip] {cities_path} exists")
    else:
        try:
            feats = overpass_top_cities(args.topn)
        except Exception as e:
            print(f"[WARN] Overpass failed ({e}); using fallback list.")
            feats = fallback_cities(args.topn)
        save_geojson(cities_path, feats)

    print("Done.")

if __name__ == "__main__":
    main()
