# scripts/fetch_osm_solar.py
# --------------------------------------------------------------
# Descarcă din Overpass (OSM) situri fotovoltaice din România
# (power=plant cu plant:source=solar sau generator:source=solar).
# Salvează GeoJSON: data/labels/ro_solar_sites.geojson
# --------------------------------------------------------------

import argparse
import json
from pathlib import Path
import requests
import geopandas as gpd
from shapely.geometry import shape, Point, Polygon, MultiPolygon, mapping

OVERPASS = "https://overpass-api.de/api/interpreter"

Q = """
[out:json][timeout:60];
area["ISO3166-1"="RO"]->.a;
(
  relation["power"="plant"]["plant:source"="solar"](area.a);
  relation["generator:source"="solar"](area.a);
  way["power"="plant"]["plant:source"="solar"](area.a);
  way["generator:source"="solar"](area.a);
  node["power"="plant"]["plant:source"="solar"](area.a);
  node["generator:source"="solar"](area.a);
);
out center geom;
"""

def main():
    p = argparse.ArgumentParser(description="Fetch Romanian PV sites from OSM (Overpass).")
    p.add_argument("--out", default="data/labels/ro_solar_sites.geojson")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    r = requests.post(OVERPASS, data=Q.encode("utf-8"))
    r.raise_for_status()
    data = r.json()

    feats = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "solar_plant")
        props = dict(name=name, osm_type=el.get("type"), id=el.get("id"), tags=tags)

        geom = None
        if "geometry" in el:  # way/relation cu coordonate
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 3:
                # încercăm poligon (nu toate sunt închise; închidem manual)
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                try:
                    geom = Polygon(coords)
                except Exception:
                    geom = None
        if geom is None:
            # fallback la punct (node sau center)
            if "lon" in el and "lat" in el:
                geom = Point(el["lon"], el["lat"])
            elif "center" in el:
                c = el["center"]
                geom = Point(c["lon"], c["lat"])

        if geom is not None:
            feats.append(dict(type="Feature", properties=props, geometry=mapping(geom)))

    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    gdf.to_file(args.out, driver="GeoJSON")
    print(f"✔ Saved {len(gdf)} features → {args.out}")

if __name__ == "__main__":
    main()
