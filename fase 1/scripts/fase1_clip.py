# ============================================================
# fase1_clip.py
# Phase 1 – Step 2: Clip the full stack to the Armero ROI
# ============================================================

import rasterio
from rasterio.mask import mask
from pyproj import Transformer

# ── EDIT THESE TWO PATHS ────────────────────────────────────
input_path  = "C:/Users/valeh/Downloads/proyecto-emergentes/armero_stack.tif"
output_path = "C:/Users/valeh/Downloads/proyecto-emergentes/armero_recortado.tif"
# ────────────────────────────────────────────────────────────

# ROI polygon in WGS84 (EPSG:4326)
roi_wgs84 = {
    "type": "Polygon",
    "coordinates": [[
        [-74.925728, 5.086328],
        [-74.834747, 5.086328],
        [-74.834747, 4.997067],
        [-74.925728, 4.997067],
        [-74.925728, 5.086328]
    ]]
}

with rasterio.open(input_path) as src:
    # Reproject polygon from WGS84 to the raster's CRS (UTM 18N)
    epsg = src.crs.to_epsg()
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    coords_proj = [
        transformer.transform(lon, lat)
        for lon, lat in roi_wgs84["coordinates"][0]
    ]

    roi_proj = {"type": "Polygon", "coordinates": [coords_proj]}

    # Clip raster to ROI
    out_image, out_transform = mask(src, [roi_proj], crop=True)

    out_meta = src.meta.copy()
    out_meta.update({
        "height":    out_image.shape[1],
        "width":     out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

print("Clipped raster saved:", output_path)
print(f"Dimensions: {out_image.shape[1]} rows x {out_image.shape[2]} cols")
print(f"Bands: {out_image.shape[0]}")
