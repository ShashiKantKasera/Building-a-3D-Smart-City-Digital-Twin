import os
import math
import html
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Hyderabad 3D Smart City Digital Twin",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
COMBINED_FILE = os.path.join(DATA_DIR, "hyderabad_city_data.geojson")
BUILDINGS_FILE = os.path.join(PROCESSED_DIR, "buildings.geojson")
ROADS_FILE = os.path.join(PROCESSED_DIR, "roads.geojson")
GREEN_FILE = os.path.join(PROCESSED_DIR, "green_spaces.geojson")

# Approximate center for the Gachibowli demonstration area
CENTER_LAT = 17.4400
CENTER_LON = 78.3480

st.title("🏙️ Hyderabad 3D Smart City Digital Twin")
st.caption(
    "Gachibowli, Hyderabad | OpenStreetMap geography | Simulated smart-city sensors"
)

# Keep a versioned map key so the map's current selection can be cleared.
if "map_key_version" not in st.session_state:
    st.session_state.map_key_version = 0
if "selected_city_feature" not in st.session_state:
    st.session_state.selected_city_feature = None

# ============================================================
# 2. GENERAL HELPERS
# ============================================================
def is_present(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() not in {"", "none", "nan", "null"}


def get_number(value):
    if not is_present(value):
        return None
    try:
        text = str(value).strip().lower()
        for unit in ("meters", "metres", "meter", "metre"):
            text = text.replace(unit, "")
        text = text.replace("m", "").strip()
        number = float(text)
        if math.isfinite(number) and number > 0:
            return number
    except (TypeError, ValueError):
        pass
    return None


def get_building_height(row):
    height = get_number(row.get("height"))
    if height is not None:
        return min(height, 300.0), "Mapped height"
    levels = get_number(row.get("building:levels"))
    if levels is not None:
        return min(levels * 3.2, 300.0), "Estimated from levels"
    return 12.0, "Fallback estimate"


def geometry_to_polygons(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def polygon_coordinates(geometry):
    if geometry is None or geometry.is_empty or geometry.geom_type != "Polygon":
        return None
    coords = [[float(x), float(y)] for x, y in geometry.exterior.coords]
    return coords if len(coords) >= 4 else None


def line_coordinates(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        lines = [geometry]
    elif geometry.geom_type == "MultiLineString":
        lines = list(geometry.geoms)
    else:
        return []
    result = []
    for line in lines:
        coords = [[float(x), float(y)] for x, y in line.coords]
        if len(coords) >= 2:
            result.append(coords)
    return result


def classify_green_space(row):
    for field in ("leisure", "landuse", "natural"):
        value = str(row.get(field, "")).lower()
        if value in {"park", "garden", "grass", "forest", "wood", "recreation_ground"}:
            return value.replace("_", " ").title()
    return "Green space"


def category_color(category):
    colors = {
        "Residential": [90, 155, 220, 220],
        "Commercial": [245, 165, 65, 230],
        "Office": [145, 110, 220, 230],
        "Industrial": [180, 125, 100, 230],
        "Public Facility": [60, 175, 135, 230],
        "Other": [150, 155, 165, 220],
    }
    return colors.get(category, colors["Other"])


def safe_text(value):
    """Convert a value to safe, compact HTML text for the popup."""
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, dict, tuple)):
        return html.escape(str(value)[:140])
    return html.escape(str(value)[:220])


def identify_feature(properties):
    """Infer the selected feature class from the picked layer properties."""
    if "building_id" in properties or "height_m" in properties:
        return "Building", "🏢"
    if "road_id" in properties or "road_type" in properties:
        return "Road", "🛣️"
    if "green_id" in properties or "green_type" in properties:
        return "Green space", "🌳"
    if "sensor_id" in properties or "sensor_type" in properties:
        return "Smart sensor", "📡"
    return "Map feature", "📍"


def extract_selected_feature(event):
    """Support Streamlit's selection state object and dict-like variants."""
    if event is None:
        return None
    try:
        selection = event.selection
    except Exception:
        selection = event.get("selection", event) if isinstance(event, dict) else event

    try:
        objects = selection.objects
    except Exception:
        objects = selection.get("objects", {}) if isinstance(selection, dict) else {}

    if not isinstance(objects, dict):
        return None

    # With single-object selection, the first object from the selected layer is enough.
    for layer_objects in objects.values():
        if isinstance(layer_objects, list) and layer_objects:
            picked = layer_objects[0]
            if isinstance(picked, dict):
                return picked
    return None


def render_feature_popup(properties):
    feature_type, icon = identify_feature(properties)

    # Keep the most useful fields near the top, then include other attributes.
    preferred = {
        "Building": ["building_id", "category", "height_m", "height_source", "floors", "lon", "lat"],
        "Road": ["road_id", "road_type"],
        "Green space": ["green_id", "green_type", "area_m2"],
        "Smart sensor": ["sensor_id", "sensor_type", "traffic_index", "aqi_demo", "temperature_c", "water_flow_lpm", "lon", "lat"],
    }.get(feature_type, [])

    ignored = {"polygon", "path", "color", "index"}
    ordered_keys = [key for key in preferred if key in properties]
    ordered_keys += [
        key for key in properties
        if key not in ordered_keys and key not in ignored and not key.startswith("__")
    ]
    ordered_keys = ordered_keys[:12]

    if ordered_keys:
        rows = "".join(
            f"<div class='twin-popup-row'><span>{html.escape(str(key).replace('_', ' ').title())}</span>"
            f"<strong>{safe_text(properties.get(key))}</strong></div>"
            for key in ordered_keys
        )
    else:
        rows = "<div class='twin-popup-row'>No attributes were returned for this selection.</div>"

    popup_html = f"""
    <div class="twin-feature-popup">
      <div class="twin-popup-header">
        <div>
          <div class="twin-popup-eyebrow">SELECTED MAP FEATURE</div>
          <div class="twin-popup-title">{icon} {html.escape(feature_type)} details</div>
        </div>
        <div class="twin-popup-tag">Selected</div>
      </div>
      <div class="twin-popup-content">{rows}</div>
      <div class="twin-popup-footer">Click another building, road, park, or sensor to inspect it.</div>
    </div>
    <style>
      .twin-feature-popup {{
        position: fixed;
        right: 24px;
        bottom: 24px;
        width: 350px;
        max-width: calc(100vw - 36px);
        max-height: 48vh;
        overflow-y: auto;
        z-index: 2147483000;
        background: rgba(255,255,255,0.98);
        color: #172033;
        border: 1px solid #d7deea;
        border-radius: 14px;
        box-shadow: 0 12px 38px rgba(15,23,42,0.28);
        font-family: sans-serif;
      }}
      .twin-popup-header {{
        display: flex; align-items: flex-start; justify-content: space-between;
        gap: 12px; padding: 15px 16px 12px; border-bottom: 1px solid #e7ebf2;
        background: #f8fafc; border-radius: 14px 14px 0 0;
      }}
      .twin-popup-eyebrow {{ font-size: 10px; letter-spacing: 1.1px; color: #64748b; font-weight: 700; margin-bottom: 5px; }}
      .twin-popup-title {{ font-size: 17px; font-weight: 750; color: #111827; }}
      .twin-popup-tag {{ font-size: 10px; color: #166534; background: #dcfce7; padding: 5px 8px; border-radius: 999px; white-space: nowrap; }}
      .twin-popup-content {{ padding: 7px 16px 9px; }}
      .twin-popup-row {{ display: flex; justify-content: space-between; gap: 14px; padding: 8px 0; border-bottom: 1px solid #eef1f5; font-size: 12px; }}
      .twin-popup-row span {{ color: #64748b; flex: 0 0 43%; }}
      .twin-popup-row strong {{ color: #1f2937; text-align: right; font-weight: 650; overflow-wrap: anywhere; }}
      .twin-popup-footer {{ padding: 10px 58px 13px 16px; color: #64748b; font-size: 11px; background: #fafbfc; border-radius: 0 0 14px 14px; min-height: 42px; box-sizing: border-box; }}
      /* The Streamlit close button is positioned over the popup footer. */
      .st-key-popup_close_button_container {{
        position: fixed !important;
        right: 29px !important;
        bottom: 27px !important;
        z-index: 2147483001 !important;
        width: 34px !important;
        min-width: 34px !important;
        margin: 0 !important;
        padding: 0 !important;
      }}
      .st-key-popup_close_button_container button {{
        width: 32px !important;
        min-width: 32px !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #334155 !important;
        font-size: 17px !important;
        line-height: 1 !important;
        box-shadow: 0 2px 8px rgba(15,23,42,0.16) !important;
      }}
      .st-key-popup_close_button_container button:hover {{
        background: #fee2e2 !important;
        border-color: #fca5a5 !important;
        color: #b91c1c !important;
      }}
      @media (max-width: 600px) {{
        .twin-feature-popup {{ right: 10px; bottom: 10px; width: calc(100vw - 20px); max-height: 42vh; }}
        .st-key-popup_close_button_container {{ right: 14px !important; bottom: 13px !important; }}
      }}
    </style>
    """
    st.markdown(popup_html, unsafe_allow_html=True)


# ============================================================
# 3. SYNTHETIC DATA FALLBACKS
# ============================================================
@st.cache_data
def create_demo_buildings():
    rng = np.random.default_rng(42)
    records = []
    categories = ["Residential", "Commercial", "Office", "Public Facility"]
    for i in range(180):
        row = i // 15
        col = i % 15
        lon = CENTER_LON + (col - 7) * 0.00048
        lat = CENTER_LAT + (row - 5.5) * 0.00048
        width = rng.uniform(0.00018, 0.00030)
        depth = rng.uniform(0.00018, 0.00030)
        height = int(rng.choice([8, 12, 18, 24, 35, 50, 75], p=[0.10, 0.16, 0.22, 0.20, 0.16, 0.10, 0.06]))
        polygon = [
            [lon - width / 2, lat - depth / 2],
            [lon + width / 2, lat - depth / 2],
            [lon + width / 2, lat + depth / 2],
            [lon - width / 2, lat + depth / 2],
        ]
        category = categories[i % len(categories)]
        records.append({
            "building_id": f"DEMO-B{i + 1:03d}", "category": category,
            "height_m": height, "height_source": "Synthetic",
            "floors": max(1, round(height / 3.2)), "lon": lon, "lat": lat,
            "polygon": polygon, "color": category_color(category),
        })
    return pd.DataFrame(records)


@st.cache_data
def create_demo_roads():
    records = []
    for i in range(13):
        lon = CENTER_LON - 0.003 + i * 0.0005
        records.append({"road_id": f"DEMO-RV{i:03d}", "road_type": "Local road", "path": [[lon, CENTER_LAT - 0.003], [lon, CENTER_LAT + 0.003]]})
    for i in range(13):
        lat = CENTER_LAT - 0.003 + i * 0.0005
        records.append({"road_id": f"DEMO-RH{i:03d}", "road_type": "Local road", "path": [[CENTER_LON - 0.003, lat], [CENTER_LON + 0.003, lat]]})
    return pd.DataFrame(records)


@st.cache_data
def create_demo_green_spaces():
    records = []
    for i in range(5):
        lon = CENTER_LON + (i - 2) * 0.0015
        lat = CENTER_LAT + (i % 2 - 0.5) * 0.002
        width, depth = 0.0005, 0.0004
        polygon = [
            [lon - width / 2, lat - depth / 2],
            [lon + width / 2, lat - depth / 2],
            [lon + width / 2, lat + depth / 2],
            [lon - width / 2, lat + depth / 2],
        ]
        records.append({"green_id": f"DEMO-G{i + 1:03d}", "green_type": "Synthetic park", "polygon": polygon, "area_m2": 0.0})
    return pd.DataFrame(records)


@st.cache_data
def create_demo_sensors():
    rng = np.random.default_rng(17)
    types = ["Traffic", "Air Quality", "Temperature", "Water Flow"]
    records = []
    for i in range(48):
        kind = types[i % len(types)]
        records.append({
            "sensor_id": f"S{i + 1:03d}", "sensor_type": kind,
            "lat": CENTER_LAT + rng.uniform(-0.003, 0.003),
            "lon": CENTER_LON + rng.uniform(-0.003, 0.003),
            "traffic_index": int(rng.integers(10, 101)),
            "aqi_demo": int(rng.integers(25, 221)),
            "temperature_c": round(float(rng.uniform(27, 42)), 1),
            "water_flow_lpm": int(rng.integers(20, 151)),
        })
    return pd.DataFrame(records)


# ============================================================
# 4. LOAD REAL GEOJSON DATA
# ============================================================
@st.cache_data
def load_buildings(path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        return pd.DataFrame()
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    records = []
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        polygons = geometry_to_polygons(geometry)
        if not polygons:
            continue
        polygon = max(polygons, key=lambda geom: geom.area)
        coords = polygon_coordinates(polygon)
        if coords is None:
            continue
        height, source = get_building_height(row)
        raw_category = row.get("building")
        category_text = str(raw_category).strip().lower() if is_present(raw_category) else "other"
        if category_text in {"apartments", "house", "residential", "detached", "terrace", "semidetached_house", "dormitory"}:
            category = "Residential"
        elif category_text in {"commercial", "retail", "supermarket", "kiosk"}:
            category = "Commercial"
        elif category_text == "office":
            category = "Office"
        elif category_text in {"industrial", "warehouse", "factory"}:
            category = "Industrial"
        elif category_text in {"school", "hospital", "public", "civic", "government", "university"}:
            category = "Public Facility"
        else:
            category = "Other"
        records.append({
            "building_id": str(row.get("id", idx)), "category": category,
            "height_m": float(height), "height_source": source,
            "floors": max(1, round(height / 3.2)), "lon": polygon.centroid.x,
            "lat": polygon.centroid.y, "polygon": coords,
            "color": category_color(category),
        })
    return pd.DataFrame(records)


@st.cache_data
def load_roads(path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        return pd.DataFrame()
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    records = []
    for idx, row in gdf.iterrows():
        for part_no, coords in enumerate(line_coordinates(row.geometry)):
            highway = row.get("highway")
            records.append({
                "road_id": f"R{idx}-{part_no}",
                "road_type": str(highway) if is_present(highway) else "Road",
                "path": coords,
            })
    return pd.DataFrame(records)


@st.cache_data
def load_green_spaces(path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        return pd.DataFrame()
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")
    projected = gdf.to_crs("EPSG:32644")
    records = []
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        parts = geometry_to_polygons(geometry)
        if not parts:
            continue
        area_geom = projected.loc[idx].geometry
        projected_parts = geometry_to_polygons(area_geom)
        for part_no, polygon in enumerate(parts):
            coords = polygon_coordinates(polygon)
            if coords is None:
                continue
            area = projected_parts[part_no].area if part_no < len(projected_parts) else area_geom.area
            records.append({
                "green_id": f"G{idx}-{part_no}",
                "green_type": classify_green_space(row),
                "polygon": coords,
                "area_m2": float(area),
            })
    return pd.DataFrame(records)


# ============================================================
# 5. SELECT DATA SOURCE
# ============================================================
st.sidebar.header("Data sources")
use_synthetic = st.sidebar.checkbox(
    "Use synthetic demo data", value=False,
    help="Use generated buildings, roads, parks and sensors for testing."
)

if use_synthetic:
    buildings = create_demo_buildings()
    roads = create_demo_roads()
    green_spaces = create_demo_green_spaces()
    data_mode = "Synthetic demo data"
else:
    try:
        if os.path.exists(BUILDINGS_FILE):
            buildings = load_buildings(BUILDINGS_FILE)
        elif os.path.exists(COMBINED_FILE):
            buildings = load_buildings(COMBINED_FILE)
        else:
            buildings = create_demo_buildings()

        if os.path.exists(ROADS_FILE):
            roads = load_roads(ROADS_FILE)
        elif os.path.exists(COMBINED_FILE):
            roads = load_roads(COMBINED_FILE)
        else:
            roads = create_demo_roads()

        if os.path.exists(GREEN_FILE):
            green_spaces = load_green_spaces(GREEN_FILE)
        elif os.path.exists(COMBINED_FILE):
            green_spaces = load_green_spaces(COMBINED_FILE)
        else:
            green_spaces = create_demo_green_spaces()

        data_mode = "OpenStreetMap data with demo fallbacks"
    except Exception as exc:
        st.error(f"Could not load real GIS data: {exc}")
        st.info("Check your GeoJSON files and CRS. Synthetic data will be used for this run.")
        buildings = create_demo_buildings()
        roads = create_demo_roads()
        green_spaces = create_demo_green_spaces()
        data_mode = "Synthetic demo data (fallback)"

if buildings.empty:
    buildings = create_demo_buildings()
if roads.empty:
    roads = create_demo_roads()
if green_spaces.empty:
    green_spaces = create_demo_green_spaces()
sensors = create_demo_sensors()
st.sidebar.caption(f"Mode: {data_mode}")

# ============================================================
# 6. FILTERS
# ============================================================
st.sidebar.header("Map layers")
show_buildings = st.sidebar.checkbox("Show buildings", value=True)
show_roads = st.sidebar.checkbox("Show roads", value=True)
show_green = st.sidebar.checkbox("Show green spaces", value=True)
show_sensors = st.sidebar.checkbox("Show smart sensors", value=True)

selected_categories = st.sidebar.multiselect(
    "Building categories",
    sorted(buildings["category"].dropna().unique()),
    default=sorted(buildings["category"].dropna().unique()),
)
min_height = st.sidebar.slider("Minimum building height (m)", 0, 100, 0)
sensor_type = st.sidebar.selectbox("Sensor type", ["All", "Traffic", "Air Quality", "Temperature", "Water Flow"])
traffic_change = st.sidebar.slider("Traffic scenario change (%)", -50, 100, 0)
green_change = st.sidebar.slider("Hypothetical green-space change (%)", -50, 100, 10)

visible_buildings = buildings[
    buildings["category"].isin(selected_categories) & (buildings["height_m"] >= min_height)
].copy()
visible_roads = roads.copy()
visible_green = green_spaces.copy()
visible_sensors = sensors.copy()
if sensor_type != "All":
    visible_sensors = visible_sensors[visible_sensors["sensor_type"] == sensor_type]

# ============================================================
# 7. KPIs
# ============================================================
total_buildings = len(visible_buildings)
total_roads = len(visible_roads)
total_green_areas = len(visible_green)
avg_height = visible_buildings["height_m"].mean() if total_buildings else 0
total_floors = visible_buildings["floors"].sum() if total_buildings else 0
mapped_height_count = int(visible_buildings["height_source"].eq("Mapped height").sum()) if "height_source" in visible_buildings.columns else 0
estimated_height_count = total_buildings - mapped_height_count
green_area_m2 = visible_green["area_m2"].sum() if "area_m2" in visible_green.columns else 0
traffic_scenario = sensors["traffic_index"].mean() * (1 + traffic_change / 100)
green_scenario = max(0, 22.0 * (1 + green_change / 100))

st.info(
    f"Data status: {data_mode}. Traffic, AQI, temperature and water-flow readings are simulated. "
    "Green-cover scenario is illustrative, not a measured percentage of the study area."
)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Buildings", f"{total_buildings:,}")
k2.metric("Road features", f"{total_roads:,}")
k3.metric("Green polygons", f"{total_green_areas:,}")
k4.metric("Average height", f"{avg_height:.1f} m")
k5.metric("Total floors (estimated)", f"{total_floors:,}")
st.caption(
    f"Building heights: {mapped_height_count:,} mapped; {estimated_height_count:,} estimated or fallback. "
    f"Green polygon area total: {green_area_m2:,.0f} m² (overlapping polygons may be counted twice)."
)

# ============================================================
# 8. CREATE MAP LAYERS (all selectable)
# ============================================================
layers = []
if show_buildings and not visible_buildings.empty:
    layers.append(pdk.Layer(
        "PolygonLayer", id="buildings-layer", data=visible_buildings,
        get_polygon="polygon", get_elevation="height_m", elevation_scale=1,
        extruded=True, wireframe=False, get_fill_color="color",
        get_line_color=[50, 60, 75, 220], line_width_min_pixels=1,
        pickable=True, auto_highlight=True,
    ))

if show_roads and not visible_roads.empty:
    layers.append(pdk.Layer(
        "PathLayer", id="roads-layer", data=visible_roads,
        get_path="path", get_color=[75, 80, 88, 230], get_width=4,
        width_min_pixels=2, width_max_pixels=6,
        pickable=True, auto_highlight=True,
    ))

if show_green and not visible_green.empty:
    layers.append(pdk.Layer(
        "PolygonLayer", id="green-spaces-layer", data=visible_green,
        get_polygon="polygon", get_elevation=1, elevation_scale=1,
        extruded=False, get_fill_color=[45, 170, 90, 175],
        get_line_color=[25, 110, 55, 230], line_width_min_pixels=1,
        pickable=True, auto_highlight=True,
    ))

sensor_colors = {
    "Traffic": [235, 65, 65, 235],
    "Air Quality": [245, 145, 35, 235],
    "Temperature": [155, 75, 220, 235],
    "Water Flow": [35, 145, 235, 235],
}
if show_sensors and not visible_sensors.empty:
    visible_sensors = visible_sensors.copy()
    visible_sensors["color"] = visible_sensors["sensor_type"].map(sensor_colors)
    layers.append(pdk.Layer(
        "ScatterplotLayer", id="sensors-layer", data=visible_sensors,
        get_position="[lon, lat]", get_radius=25,
        radius_min_pixels=5, radius_max_pixels=12,
        get_fill_color="color", get_line_color=[255, 255, 255, 255],
        line_width_min_pixels=1, stroked=True,
        pickable=True, auto_highlight=True,
    ))

# ============================================================
# 9. INTERACTIVE MAP + CLICK SELECTION
# ============================================================
all_lons = [CENTER_LON]
all_lats = [CENTER_LAT]
if not visible_buildings.empty:
    all_lons.extend(visible_buildings["lon"].tolist())
    all_lats.extend(visible_buildings["lat"].tolist())
if show_sensors and not visible_sensors.empty:
    all_lons.extend(visible_sensors["lon"].tolist())
    all_lats.extend(visible_sensors["lat"].tolist())

view_state = pdk.ViewState(
    latitude=float(np.mean(all_lats)),
    longitude=float(np.mean(all_lons)),
    zoom=14,
    pitch=55,
    bearing=-20,
)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={
        "html": "<b>Click feature to see details in the bottom-right panel</b>",
        "style": {"backgroundColor": "#202b3c", "color": "white"},
    },
)

st.subheader("Interactive 3D urban model")
st.caption("Click any building, road, green-space polygon, or sensor to open its details in the bottom-right popup.")

# The selection API is available in recent Streamlit versions.
map_event = st.pydeck_chart(
    deck,
    height=680,
    width="stretch",
    on_select="rerun",
    selection_mode="single-object",
    key=f"city_twin_map_{st.session_state.map_key_version}",
)

picked = extract_selected_feature(map_event)
if picked:
    st.session_state.selected_city_feature = picked

# A normal Streamlit button clears the selection and remounts the map.
clear_col, note_col = st.columns([1, 5])
with clear_col:
    if st.button("Clear selected feature", key="clear_city_feature"):
        st.session_state.selected_city_feature = None
        st.session_state.map_key_version += 1
        st.rerun()
with note_col:
    st.caption("The popup is fixed to the bottom-right of the browser window. Select another feature to update it.")

if st.session_state.selected_city_feature:
    # Close control overlays the lower-right corner of the fixed popup.
    with st.container(key="popup_close_button_container"):
        if st.button("×", key="close_feature_popup", help="Close feature details popup"):
            st.session_state.selected_city_feature = None
            st.session_state.map_key_version += 1
            st.rerun()

    render_feature_popup(st.session_state.selected_city_feature)

st.caption(
    "Map features use OpenStreetMap data where available. Some building heights are estimated. "
    "Sensor readings and planning scenarios are synthetic. OSM attribution: © OpenStreetMap contributors."
)

# ============================================================
# 10. SENSOR ANALYTICS AND ALERTS
# ============================================================
left, right = st.columns(2)
with left:
    st.subheader("Smart sensor summary")
    sensor_summary = sensors.groupby("sensor_type").agg(
        sensor_count=("sensor_id", "count"),
        avg_traffic_index=("traffic_index", "mean"),
        avg_demo_aqi=("aqi_demo", "mean"),
        avg_temperature_c=("temperature_c", "mean"),
        avg_water_flow_lpm=("water_flow_lpm", "mean"),
    ).round(1)
    st.dataframe(sensor_summary, width="stretch")
with right:
    st.subheader("Simulated alerts")
    st.metric("High traffic sensors", int((sensors["traffic_index"] >= 75).sum()))
    st.metric("High demo AQI sensors", int((sensors["aqi_demo"] >= 150).sum()))
    st.metric("High temperature sensors", int((sensors["temperature_c"] >= 38).sum()))
    st.caption("Demo thresholds only; these are not official Hyderabad monitoring readings.")

# ============================================================
# 11. SCENARIOS AND DATA INSPECTION
# ============================================================
st.subheader("Urban planning scenario")
s1, s2, s3 = st.columns(3)
s1.metric("Traffic scenario index", f"{traffic_scenario:.1f}", f"{traffic_change:+d}%")
s2.metric("Green-space scenario index", f"{green_scenario:.1f}%", f"{green_change:+d}%")
s3.metric("Visible average building height", f"{avg_height:.1f} m")
st.write(
    "Scenario sliders apply simple mathematical assumptions. They do not represent validated traffic, "
    "air-quality, or land-cover predictions."
)

with st.expander("Inspect building records"):
    if not visible_buildings.empty:
        cols = [c for c in ["building_id", "category", "height_m", "height_source", "floors", "lon", "lat"] if c in visible_buildings.columns]
        st.dataframe(visible_buildings[cols], width="stretch")
    else:
        st.write("No buildings match the selected filters.")

with st.expander("Inspect road records"):
    if not visible_roads.empty:
        st.dataframe(visible_roads.drop(columns=["path"], errors="ignore"), width="stretch")
    else:
        st.write("No road records available.")

with st.expander("Inspect green-space records"):
    if not visible_green.empty:
        st.dataframe(visible_green.drop(columns=["polygon"], errors="ignore"), width="stretch")
    else:
        st.write("No green-space records available.")

with st.expander("Data sources and limitations"):
    st.markdown(
        """
        **Geographic data**
        - OpenStreetMap building footprints, roads and mapped green spaces.
        - Mapped building heights are used when available; missing heights are estimated.

        **Synthetic data**
        - Traffic index, air quality, temperature and water flow.
        - Synthetic fallback buildings, roads and parks when source files are absent or unusable.
        - Planning scenario indicators.

        **Limitations**
        - OpenStreetMap coverage varies by location.
        - Green-space polygons can overlap or describe different types of land use.
        - This prototype does not ingest live traffic or environmental sensor feeds.
        """
    )

st.divider()
st.caption("Hyderabad 3D Smart City Digital Twin | Python + Streamlit + GeoPandas + PyDeck | © OpenStreetMap contributors")
