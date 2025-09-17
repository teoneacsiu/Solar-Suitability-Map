# scripts/visualize.py
# --------------------------------------------------------------
# Harta interactivă (HTML) + PNG color pentru un GeoTIFF.
# Opțional: suprapun straturi de “condiții” (apă/urban/pădure/pantă),
# re-proiectate pe grila hărții principale ca să nu existe mismatch.
# Panourile UI sunt poziționate să NU se suprapună:
#  - Legendă (stânga-sus, îngustă)
#  - About (stânga-jos, îngust, cu linkuri clicabile)
#  - Area summary (mijloc-stânga, prin buton toggle)
#  - Model evaluation (dreapta-jos, dacă --metrics)
#  - Suitability zones (poligoane postprocesate, dacă --zones)
# Suprafața se calculează în metri, robust la CRS (și când e în grade).
# --------------------------------------------------------------

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import array_bounds
from PIL import Image
import folium

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROC_DIR = ROOT / "data" / "processed"

BASEMAPS = {
    "CartoDB.Positron": dict(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="© OpenStreetMap, © CARTO",
        name="CartoDB.Positron",
        max_zoom=20,
    ),
    "OpenStreetMap": dict(tiles="OpenStreetMap", name="OpenStreetMap"),
    "Esri.WorldImagery": dict(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, and the GIS User Community",
        name="Esri.WorldImagery",
        max_zoom=20,
    ),
}

ESRI_BOUNDARIES = dict(
    tiles="https://services.arcgisonline.com/arcgis/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Boundaries & Places",
    overlay=True,
    control=True,
)

# ---------- utilitare culoare / PNG ----------


def to_uint8(x, vmin=None, vmax=None):
    x = x.astype("float32")
    if vmin is None:
        vmin = np.nanpercentile(x, 2)
    if vmax is None:
        vmax = np.nanpercentile(x, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(x), np.nanmax(x)
    x = np.clip((x - vmin) / (vmax - vmin + 1e-9), 0, 1)
    return (x * 255).astype(np.uint8)


def rgba_from_prob(prob, nodata_mask=None):
    u = to_uint8(prob)
    r = 255 - u
    g = u
    b = np.zeros_like(u)
    a = np.full_like(u, 255, dtype=np.uint8)
    rgba = np.dstack([r, g, b, a])
    if nodata_mask is not None:
        rgba[nodata_mask] = [240, 200, 40, 255]
    return rgba


def rgba_from_classes(arr, threshold=0.5, nodata_mask=None):
    cls = (arr >= float(threshold)).astype(np.uint8)
    h, w = cls.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[cls == 1] = [0, 170, 0, 255]
    rgba[cls == 0] = [220, 40, 40, 255]
    if nodata_mask is not None:
        rgba[nodata_mask] = [240, 200, 40, 255]
    return rgba


def save_png_rgba(rgba, out_path, max_side=4096):
    H, W = rgba.shape[:2]
    if max(H, W) > max_side:
        scale = max_side / max(H, W)
        new_size = (int(W * scale), int(H * scale))
        img = Image.fromarray(rgba).resize(new_size, resample=Image.BILINEAR)
    else:
        img = Image.fromarray(rgba)
    Image.fromarray(np.array(img)).save(out_path)


# ---------- vector overlays (judete/orase) ----------


def add_geojson(m, path, name, style=None, tooltip_field=None, tooltip_fields=None, aliases=None, show=True):
    try:
        import geopandas as gpd
    except Exception as e:
        print(f"[WARN] '{name}' skipped (geopandas missing): {e}")
        return
    gdf = gpd.read_file(path)
    gj = json.loads(gdf.to_json())
    styfn = (lambda _: style) if style else None

    # Tooltip (hover): fie un câmp, fie mai multe câmpuri
    tooltip = None
    if tooltip_fields:
        existing = [f for f in tooltip_fields if f in gdf.columns]
        if existing:
            from folium.features import GeoJsonTooltip
            tooltip = GeoJsonTooltip(fields=existing, aliases=aliases or existing, sticky=False, labels=True)
    elif tooltip_field and tooltip_field in gdf.columns:
        from folium.features import GeoJsonTooltip
        tooltip = GeoJsonTooltip(fields=[tooltip_field], aliases=[tooltip_field], sticky=False, labels=True)

    folium.GeoJson(gj, name=name, style_function=styfn, tooltip=tooltip, show=show).add_to(m)



# ---------- condiții (din sentinel + slope) ----------


def compute_indices_from_sentinel(s2_path):
    """Returnează NDVI/NDBI/NDWI(aprox) + transform/crs din compozitul S2 [B2,B4,B8,B11]."""
    with rasterio.open(s2_path) as src:
        B2, B4, B8, B11 = src.read().astype("float32")
        transform = src.transform
        crs = src.crs
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    ndbi = (B11 - B8) / (B11 + B8 + 1e-6)
    ndwi = (B2 - B11) / (B2 + B11 + 1e-6)
    return dict(ndvi=ndvi, ndbi=ndbi, ndwi=ndwi, transform=transform, crs=crs)


def reproject_to_ref(
    src_arr, src_transform, src_crs, ref_transform, ref_crs, ref_shape, resampling=Resampling.bilinear
):
    """Reproiectează un array pe grila de referință (forma/transform/crs ale hărții)."""
    dst = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=resampling,
    )
    return dst


def make_mask_rgba(mask, color_rgb, alpha=130):
    h, w = mask.shape
    r, g, b = color_rgb
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = np.where(mask, alpha, 0).astype(np.uint8)
    return rgba


def safe_binary_dilation(mask, radius_px):
    if radius_px <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation

        structure = np.ones((2 * radius_px + 1, 2 * radius_px + 1), dtype=bool)
        return binary_dilation(mask, structure=structure)
    except Exception as e:
        print(f"[WARN] buffer skipped (SciPy missing): {e}")
        return mask


# ---- calc. robust de arie (în metri), indiferent de CRS-ul hărții ----


def pixel_area_km2(transform, crs, shape_hw):
    """
    Suprafața unui pixel, în km².
    - Dacă CRS-ul e proiectat (metri), folosește |a*e| direct (cel mai robust).
    - Altfel, estimează din bounds reproiectate în EPSG:3857 (metri).
    """
    try:
        is_projected = getattr(crs, "is_projected", False)
    except Exception:
        is_projected = False

    if is_projected or "3857" in str(crs) or "3395" in str(crs):
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        return (px_w * px_h) / 1e6

    H, W = shape_hw
    b = array_bounds(H, W, transform)
    x0, y0, x1, y1 = transform_bounds(crs, "EPSG:3857", *b, densify_pts=21)
    px_w_m = abs((x1 - x0) / W)
    px_h_m = abs((y1 - y0) / H)
    return (px_w_m * px_h_m) / 1e6


# ---------- UI panouri ----------


def add_fixed_card(m, html, *, left=None, right=None, top=None, bottom=None, width="320px", max_height="80vh"):
    """Card HTML fix pe hartă, cu poziționare flexibilă (stânga/dreapta, sus/jos)."""
    style_bits = []
    if left is not None:
        style_bits.append(f"left:{left};")
    if right is not None:
        style_bits.append(f"right:{right};")
    if top is not None:
        style_bits.append(f"top:{top};")
    if bottom is not None:
        style_bits.append(f"bottom:{bottom};")
    style = " ".join(style_bits) or "left:10px; top:10px;"
    div = folium.Element(
        f"""
    <div style="position: fixed; z-index: 9999; {style}
        width:{width}; max-height:{max_height}; overflow:auto;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        border-radius: 6px; padding: 10px; font-family: sans-serif;
        font-size: 12px; line-height: 1.4; word-break: break-word;">
        {html}
    </div>
    """
    )
    m.get_root().html.add_child(div)


def make_legend_html():
    return """
    <b>Legend</b><br>
    <div style="margin-top:6px">
      <div><span style="display:inline-block;width:12px;height:12px;background:#d62828;margin-right:6px;"></span> Low suitability</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#00aa00;margin-right:6px;"></span> High suitability</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#f0c828;margin-right:6px;"></span> No data</div>
      <hr style="margin:8px 0">
      <div><span style="display:inline-block;width:12px;height:12px;background:rgba(0,120,255,0.7);margin-right:6px;"></span> Water</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:rgba(0,0,0,0.6);margin-right:6px;"></span> Urban / built-up</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:rgba(0,160,0,0.6);margin-right:6px;"></span> Dense forest</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:rgba(140,80,20,0.6);margin-right:6px;"></span> High slope</div>
    </div>
    """


def make_about_html():
    # Panou "About" extins, cu linkuri COMPLETE și descrieri scurte (clicabile)
    return """
    <b>About this map</b><br>
    <small>
      <p><b>Ce face harta?</b><br>
      Întreabă: „Unde în România ar fi potrivit să amplasezi un parc fotovoltaic mare?”<br>
      Combină <i>resursa solară</i>, <i>constrângerile fizice/umane</i> (pantă, apă, urban, pădure)
      și <i>starea la sol</i> (indici din Sentinel-2), apoi afișează <b>probabilitate</b>
      sau <b>clase</b> (bun / mai puțin bun).</p>

      <p><b>Date și model</b><br>
      <u>Imagini</u>: Sentinel-2 (benzi B2/B4/B8/B11). <u>Relief</u>: DEM → pantă (grade).
      <u>Climă</u>: ERA5-Land ca proxy pentru GHI/PVOUT (soare pe termen lung).
      <u>Model</u>: Random Forest pe <i>NDVI/NDBI/NDWI, slope, GHI</i>. Etichetele de antrenare sunt
      derivate din reguli (pantă mică, non-pădure, soare mare).</p>

      <p><b>Indici (pe scurt)</b><br>
      NDVI = (NIR−Red)/(NIR+Red) → vegetație; NDBI = (SWIR−NIR)/(SWIR+NIR) → construit;
      NDWI/MNDWI → apă. În compozitul nostru avem B2 (Blue), B4 (Red), B8 (NIR), B11 (SWIR1).</p>

      <p><b>Condițiile (filtrele) aplicate</b><br>
      1) <b>Panta terenului</b> – excludem pante &gt; ~7° (literatura folosește frecvent 5–10°).<br>
      2) <b>Apă</b> – NDWI/MNDWI &gt; ~0.05 ⇒ apă; aplicăm și un mic buffer (ex. 1 px).<br>
      3) <b>Urban/construit</b> – NDBI &gt; ~0.05 ⇒ construit (evităm orașele/zona densă).<br>
      4) <b>Pădure deasă</b> – NDVI &gt; ~0.55 ⇒ pădure (nu țintim defrișări).</p>

      <p><b>De ce sunt rezonabile aceste praguri?</b><br>
      Linkuri <b>complete</b> + surse:</p>

      <ul style="padding-left:16px; margin-top:6px; margin-bottom:6px">
        <li><b>Criterii PV / pante 5–10° (review IEEE)</b><br>
          <a href="https://ieeexplore.ieee.org/document/10160839" target="_blank" rel="noopener">
          https://ieeexplore.ieee.org/document/10160839</a><br>
          (IEEE Xplore – articol de sinteză despre criterii GIS pentru amplasare PV)</li>

        <li><b>Exemple praguri GIS pentru PV (MDPI Applied Sciences)</b><br>
          <a href="https://www.mdpi.com/2076-3417/14/19/8663" target="_blank" rel="noopener">
          https://www.mdpi.com/2076-3417/14/19/8663</a><br>
          (Revistă științifică open-access; pante 5–10°, excluderi standard)</li>

        <li><b>NDWI original (McFeeters, 1996)</b><br>
          <a href="https://doi.org/10.1080/01431169608948714" target="_blank" rel="noopener">
          https://doi.org/10.1080/01431169608948714</a><br>
          (International Journal of Remote Sensing – detectarea apelor)</li>

        <li><b>MNDWI (Xu, 2006) – robust în zone urbane</b><br>
          <a href="https://doi.org/10.1080/01431160600589179" target="_blank" rel="noopener">
          https://doi.org/10.1080/01431160600589179</a><br>
          (IJRS – variantă NDWI pentru orașe)</li>

        <li><b>NDBI (Zha, Ni &amp; Yang, 2003)</b><br>
          <a href="https://doi.org/10.1109/IGARSS.2003.1294665" target="_blank" rel="noopener">
          https://doi.org/10.1109/IGARSS.2003.1294665</a><br>
          (IEEE IGARSS – indice pentru zone construite)</li>

        <li><b>NDVI explicat pe înțelesul tuturor (NASA)</b><br>
          <a href="https://earthobservatory.nasa.gov/features/MeasuringVegetation" target="_blank" rel="noopener">
          https://earthobservatory.nasa.gov/features/MeasuringVegetation</a><br>
          (NASA Earth Observatory – ce e NDVI, intervale tipice)</li>

        <li><b>Hărți globale NDVI (NASA)</b><br>
          <a href="https://earthobservatory.nasa.gov/global-maps/MOD13A2_NDVI" target="_blank" rel="noopener">
          https://earthobservatory.nasa.gov/global-maps/MOD13A2_NDVI</a><br>
          (NASA – exemple vizuale de valori NDVI)</li>

        <li><b>ERA5-Land (Copernicus CDS) – date climatice pe termen lung</b><br>
          <a href="https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means" target="_blank" rel="noopener">
          https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means</a><br>
          (Copernicus Climate Data Store – medii lunare)</li>

        <li><b>DOI ERA5-Land</b><br>
          <a href="https://doi.org/10.24381/cds.e2161bac" target="_blank" rel="noopener">
          https://doi.org/10.24381/cds.e2161bac</a><br>
          (Înregistrare oficială a dataset-ului)</li>

        <li><b>PVGIS (JRC, Comisia Europeană) – documentație PVOUT</b><br>
          <a href="https://joint-research-centre.ec.europa.eu/pvgis-documentation_en" target="_blank" rel="noopener">
          https://joint-research-centre.ec.europa.eu/pvgis-documentation_en</a><br>
          (JRC – metodologie și resurse pentru PV în UE)</li>
      </ul>

      <p><b>De ce un mic buffer (1 pixel) la apă/urban?</b><br>
      Evităm pixeli „buni” lipiți de maluri/clădiri. La ~200 m/pixel, 1 px ≈ 200 m;
      poți crește la 2–3 px dacă vrei zone tampon mai mari.</p>

      <p><b>Arie (km²) corectă indiferent de CRS</b><br>
      Estimăm aria pe pixel reproiectând extinderea în EPSG:3857 (metri), astfel încât
      rezultatele nu depind dacă harta e în grade sau metri.</p>

      <p style="margin-bottom:0"><b>Interpretare rapidă</b><br>
      Strat principal: roșu=slab, verde=bun, galben=no data. Overlays: Water (albastru),
      Urban (negru), Dense forest (verde), High slope (maro). Butonul „Summary”
      (mijloc-stânga) arată km² și % pentru fiecare categorie.</p>
    </small>
    """


def make_stats_html(stats):
    def row(label, km2, pct):
        km2s = f"{km2:,.1f}".replace(",", " ")
        pcts = f"{pct:,.2f}".replace(",", " ")
        return f"<tr><td>{label}</td><td style='text-align:right'>{km2s}</td><td style='text-align:right'>{pcts}%</td></tr>"

    rows = "".join(
        [
            row("Water", stats["km2_water"], stats["pct_water"]),
            row("Urban", stats["km2_urban"], stats["pct_urban"]),
            row("Dense forest", stats["km2_forest"], stats["pct_forest"]),
            row("High slope", stats["km2_slope"], stats["pct_slope"]),
            "<tr><td colspan='3'><hr></td></tr>",
            row("<b>Candidate (kept)</b>", stats["km2_kept"], stats["pct_kept"]),
        ]
    )
    return f"""
    <b>Area summary</b>
    <table style="width:100%;font-size:12px;margin-top:6px;border-collapse:collapse">
      <thead><tr><th style="text-align:left">Layer</th><th style="text-align:right">km²</th><th style="text-align:right">%</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """


def add_summary_toggle(m, stats_html):
    """Buton „Summary” (mijloc-stânga) care deschide/închide panoul cu tabelul."""
    container = f"""
    <div id="summaryBtn" style="
        position: fixed; left: 12px; top: 45%; transform: translateY(-50%);
        z-index: 9999; background:#1976d2; color:#fff; padding:8px 10px; border-radius:4px;
        font-family:sans-serif; font-size:12px; cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,.25);">
      Summary
    </div>
    <div id="summaryPanel" style="
        position: fixed; left: 12px; top: calc(45% + 36px); transform: translateY(-50%);
        z-index: 9999; width: 280px; max-height: 60vh; overflow:auto;
        display:none; background:rgba(255,255,255,.97); padding:10px;
        border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,.25); font-family:sans-serif;">
      {stats_html}
    </div>
    <script>
      (function() {{
        var btn = document.getElementById('summaryBtn');
        var panel = document.getElementById('summaryPanel');
        btn.addEventListener('click', function() {{
          panel.style.display = (panel.style.display === 'none' || panel.style.display === '') ? 'block' : 'none';
        }});
      }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(container))


def add_metrics_card(m, metrics_path, *, right="10px", bottom="10px", width="280px"):
    """
    Panou 'Model evaluation' (dreapta-jos).
    Citește metrics.json și, dacă există, inserează roc.png / pr.png / cm.png ca <img> inline (base64).
    """
    try:
        mp = Path(metrics_path)
        if not mp.exists():
            print(f"[INFO] metrics.json not found: {mp}")
            return
        meta = json.loads(mp.read_text())

        def img_tag(p: Path):
            if p.exists():
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                return f'<img src="data:image/png;base64,{b64}" style="width:100%;border:1px solid #ddd;margin-bottom:6px"/>'
            return ""

        base = mp.parent
        html = f"""
        <b>Model evaluation</b>
        <table style="width:100%;font-size:12px;margin-top:6px;border-collapse:collapse">
          <tr><td>ROC-AUC</td><td style='text-align:right'>{meta.get('roc_auc',0):.3f}</td></tr>
          <tr><td>PR-AUC</td><td style='text-align:right'>{meta.get('pr_auc',0):.3f}</td></tr>
          <tr><td>Best th</td><td style='text-align:right'>{meta.get('best_threshold',0):.3f}</td></tr>
          <tr><td>F1 / P / R</td><td style='text-align:right'>{meta.get('f1',0):.3f} / {meta.get('precision',0):.3f} / {meta.get('recall',0):.3f}</td></tr>
        </table>
        <div style="margin-top:6px">
          {img_tag(base / 'roc.png')}
          {img_tag(base / 'pr.png')}
          {img_tag(base / 'cm.png')}
        </div>
        """
        div = folium.Element(
            f"""
        <div style="position: fixed; z-index: 9999; right:{right}; bottom:{bottom};
            width:{width}; max-height: 34vh; overflow:auto;
            background: rgba(255,255,255,0.95); box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            border-radius: 6px; padding: 10px; font-family: sans-serif; font-size: 12px;">
            {html}
        </div>"""
        )
        m.get_root().html.add_child(div)
    except Exception as e:
        print(f"[WARN] metrics panel failed: {e}")


# ---------- main ----------


def main():
    p = argparse.ArgumentParser(description="Create explainable HTML map (+PNG) with overlays.")
    p.add_argument(
        "--map",
        default=str(PROC_DIR / "suitability_adjusted.tif"),
        help="GeoTIFF to visualize (default: suitability_adjusted.tif; falls back to suitability_map.tif)",
    )
    p.add_argument(
        "--viz-mode", choices=["prob", "classes"], default="prob", help="prob = red→green gradient; classes = binary at --threshold"
    )
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--basemap", default="CartoDB.Positron")
    p.add_argument("--opacity", type=float, default=0.7)
    p.add_argument("--png-max", type=int, default=4096)
    p.add_argument("--out-suffix", dest="out_suffix", default="", help="Suffix for output filenames (e.g. 'masked')")
    p.add_argument("--out-sufix", dest="out_suffix", help=argparse.SUPPRESS)  # alias tolerant

    p.add_argument("--counties", type=str, default="", help="Path to counties GeoJSON")
    p.add_argument("--cities", type=str, default="", help="Path to cities GeoJSON")
    p.add_argument("--add-admin-tiles", action="store_true", help="Add Esri Boundaries & Places")

    p.add_argument("--constraints", action="store_true", help="Overlay water/urban/forest/slope computed from processed rasters")
    p.add_argument(
        "--s2", default=str(PROC_DIR / "sentinel_composite.tif"), help="Processed Sentinel-2 composite with bands [B2,B4,B8,B11]"
    )
    p.add_argument("--slope", default=str(PROC_DIR / "slope.tif"), help="Processed slope raster")

    p.add_argument("--slope-max", type=float, default=7.0)
    p.add_argument("--ndvi-forest", type=float, default=0.55)
    p.add_argument("--ndwi-water", type=float, default=0.05)
    p.add_argument("--ndbi-urban", type=float, default=0.05)
    p.add_argument("--buffer-water-px", type=int, default=1)
    p.add_argument("--buffer-urban-px", type=int, default=1)

    p.add_argument("--no-stats-panel", action="store_true", help="Do not render the Area summary panel")

    p.add_argument("--metrics", type=str, default="", help="Path to metrics.json to display a 'Model evaluation' card")
    p.add_argument("--zones", type=str, default="", help="Path to zones GeoJSON (polygons) to overlay")

    args = p.parse_args()

    # --- harta principală
    tif = Path(args.map)
    if not tif.exists():
        alt = PROC_DIR / "suitability_map.tif"
        if alt.exists():
            print(f"[WARN] Map not found: {tif} → using {alt}")
            tif = alt
        else:
            raise SystemExit(f"[FATAL] Missing map: {args.map}")

    with rasterio.open(tif) as src:
        arr = src.read(1).astype("float32")
        map_crs = src.crs
        map_transform = src.transform
        b = transform_bounds(map_crs, "EPSG:4326", *src.bounds)
        west, south, east, north = b
        center = [(south + north) / 2, (west + east) / 2]

    # --- colorizare PNG
    nodata_mask = ~np.isfinite(arr)
    if args.viz_mode == "classes":
        rgba = rgba_from_classes(arr, threshold=args.threshold, nodata_mask=nodata_mask)
    else:
        rgba = rgba_from_prob(arr, nodata_mask=nodata_mask)

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    png_path = PROC_DIR / f"suitability_colored{suffix}.png"
    save_png_rgba(rgba, png_path, max_side=args.png_max)
    print(f"✔ PNG color: {png_path}")

    # --- hartă folium
    m = folium.Map(location=center, zoom_start=6, control_scale=True, tiles=None)
    if args.basemap in BASEMAPS:
        folium.TileLayer(**BASEMAPS[args.basemap]).add_to(m)
    else:
        folium.TileLayer(tiles=args.basemap, name="Basemap").add_to(m)

    folium.raster_layers.ImageOverlay(
        name=f"Suitability ({args.viz_mode})",
        image=str(png_path),
        bounds=[[south, west], [north, east]],
        opacity=args.opacity,
        interactive=False,
        cross_origin=False,
        zindex=5,
        show=True,
    ).add_to(m)

    # Admin overlays
    if args.add_admin_tiles:
        folium.TileLayer(**ESRI_BOUNDARIES).add_to(m)

    if args.counties:
        add_geojson(
            m,
            args.counties,
            name="Județe",
            style={"color": "#222", "weight": 1, "fillOpacity": 0.0},
            tooltip_field="name",
            show=True,
        )
    
    if args.cities:
        add_geojson(
            m, args.cities, name="Orașe",
            style={"color": "#000", "weight": 0.5, "fillOpacity": 0.0},
            tooltip_fields=["name", "population"],  # schimbă aici dacă ai alte nume de coloane
            aliases=["Oraș", "Populație"],
            show=True,
        )


    # --- zone (poligoane) din postprocess_zones.py
    if args.zones:
        add_geojson(
            m,
            args.zones,
            name="Suitability zones",
            style={"color": "#ff6600", "weight": 1.2, "fillOpacity": 0.10},
            tooltip_field=None,
            show=True,
        )

    # --- condiții (reproiectate pe grila hărții)
    stats = None
    if args.constraints:
        s2_path = Path(args.s2)
        slope_path = Path(args.slope)
        if not s2_path.exists() or not slope_path.exists():
            print("[WARN] Constraints skipped (missing sentinel_composite.tif or slope.tif).")
        else:
            idx = compute_indices_from_sentinel(s2_path)
            ndvi_raw, ndbi_raw, ndwi_raw = idx["ndvi"], idx["ndbi"], idx["ndwi"]
            s2_tr, s2_crs = idx["transform"], idx["crs"]

            with rasterio.open(slope_path) as ssrc:
                slope_raw = ssrc.read(1).astype("float32")
                slope_tr, slope_crs = ssrc.transform, ssrc.crs

            ref_shape = arr.shape
            ndvi = reproject_to_ref(ndvi_raw, s2_tr, s2_crs, map_transform, map_crs, ref_shape, Resampling.bilinear)
            ndbi = reproject_to_ref(ndbi_raw, s2_tr, s2_crs, map_transform, map_crs, ref_shape, Resampling.bilinear)
            ndwi = reproject_to_ref(ndwi_raw, s2_tr, s2_crs, map_transform, map_crs, ref_shape, Resampling.bilinear)
            slope = reproject_to_ref(
                slope_raw, slope_tr, slope_crs, map_transform, map_crs, ref_shape, Resampling.bilinear
            )

            water_mask = ndwi > float(args.ndwi_water)
            urban_mask = ndbi > float(args.ndbi_urban)
            forest_mask = ndvi > float(args.ndvi_forest)
            slope_mask = slope > float(args.slope_max)

            water_mask_b = safe_binary_dilation(water_mask, args.buffer_water_px)
            urban_mask_b = safe_binary_dilation(urban_mask, args.buffer_urban_px)

            total_px = int(np.count_nonzero(np.isfinite(arr)))
            water_px = int(np.count_nonzero(water_mask_b))
            urban_px = int(np.count_nonzero(urban_mask_b))
            forest_px = int(np.count_nonzero(forest_mask))
            slope_px = int(np.count_nonzero(slope_mask))
            kept_mask = (~water_mask_b) & (~urban_mask_b) & (~forest_mask) & (~slope_mask) & np.isfinite(arr)
            kept_px = int(np.count_nonzero(kept_mask))

            px_km2 = pixel_area_km2(map_transform, map_crs, arr.shape)
            km2_water = water_px * px_km2
            km2_urban = urban_px * px_km2
            km2_forest = forest_px * px_km2
            km2_slope = slope_px * px_km2
            km2_kept = kept_px * px_km2

            def pct(x):
                return 100.0 * x / max(1, total_px)

            stats = dict(
                km2_water=km2_water,
                pct_water=pct(water_px),
                km2_urban=km2_urban,
                pct_urban=pct(urban_px),
                km2_forest=km2_forest,
                pct_forest=pct(forest_px),
                km2_slope=km2_slope,
                pct_slope=pct(slope_px),
                km2_kept=km2_kept,
                pct_kept=pct(kept_px),
            )

            overlays = [
                ("Water", make_mask_rgba(water_mask_b, (0, 120, 255), alpha=160)),
                ("Urban", make_mask_rgba(urban_mask_b, (0, 0, 0), alpha=150)),
                ("Dense forest", make_mask_rgba(forest_mask, (0, 160, 0), alpha=150)),
                ("High slope", make_mask_rgba(slope_mask, (140, 80, 20), alpha=150)),
            ]
            for name, rgba_mask in overlays:
                path = PROC_DIR / f"overlay_{name.replace(' ', '_').lower()}{suffix}.png"
                save_png_rgba(rgba_mask, path, max_side=args.png_max)
                folium.raster_layers.ImageOverlay(
                    name=name,
                    image=str(path),
                    bounds=[[south, west], [north, east]],
                    opacity=1.0,
                    interactive=False,
                    zindex=7,
                    show=False,
                ).add_to(m)

    # --- UI: legendă (stânga-sus) + about (stânga-jos, cu linkuri)
    add_fixed_card(m, make_legend_html(), left="10px", top="10px", width="200px", max_height="34vh")
    add_fixed_card(m, make_about_html(), left="10px", bottom="10px", width="260px", max_height="30vh")

    # --- UI: summary în mijloc-stânga, *toggle*
    if (stats is not None) and (not args.no_stats_panel):
        add_summary_toggle(m, make_stats_html(stats))

    # --- UI: model evaluation (dreapta-jos) dacă avem metrics.json
    if args.metrics:
        add_metrics_card(m, args.metrics, right="12px", bottom="12px", width="260px")

    folium.LayerControl(collapsed=False).add_to(m)

    # --- salvare HTML
    html_path = PROC_DIR / f"suitability_map{suffix}.html"
    m.save(str(html_path))
    print(f"✔ Harta interactivă: {html_path}")
    print('   Deschide în browser. Activează/dezactivează straturi din "Layer Control".')


if __name__ == "__main__":
    main()
