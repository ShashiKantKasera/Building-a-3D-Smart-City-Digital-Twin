import pandas as pd
import os
import geopandas as gpd

INPUT_FILE = "data/hyderabad_city_data.geojson"
OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

gdf = gpd.read_file(INPUT_FILE)

print("Total features:", len(gdf))
print("Available fields:", gdf.columns.tolist())

# Building footprints
if "building" in gdf.columns:
    buildings = gdf[
        gdf["building"].notna()
        & (gdf["building"].astype(str) != "")
    ].copy()
else:
    buildings = gdf.iloc[0:0].copy()

# Roads and paths
if "highway" in gdf.columns:
    roads = gdf[
        gdf["highway"].notna()
        & (gdf["highway"].astype(str) != "")
    ].copy()
else:
    roads = gdf.iloc[0:0].copy()

# Green spaces
green = gdf.iloc[0:0].copy()

for field, values in {
    "leisure": ["park", "garden", "recreation_ground"],
    "landuse": ["grass", "forest", "recreation_ground"],
    "natural": ["wood"],
}.items():
    if field in gdf.columns:
        matched = gdf[
            gdf[field].astype(str).isin(values)
        ]
        green = gpd.GeoDataFrame(
            pd.concat([green, matched]).drop_duplicates(),
            geometry=gdf.geometry.name,
            crs=gdf.crs,
        )

# Keep the original CRS; GeoJSON output will use WGS84.
if gdf.crs is not None:
    buildings = buildings.to_crs("EPSG:4326")
    roads = roads.to_crs("EPSG:4326")
    green = green.to_crs("EPSG:4326")

buildings.to_file(
    f"{OUTPUT_DIR}/buildings.geojson",
    driver="GeoJSON",
)
roads.to_file(
    f"{OUTPUT_DIR}/roads.geojson",
    driver="GeoJSON",
)
green.to_file(
    f"{OUTPUT_DIR}/green_spaces.geojson",
    driver="GeoJSON",
)

print("Buildings:", len(buildings))
print("Roads:", len(roads))
print("Green spaces:", len(green))
print("Files saved in:", OUTPUT_DIR)
