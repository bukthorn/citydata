import math
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

# Overpass and Nominatim operators ask for an identifying User-Agent with contact info
USER_AGENT = "citydata-streamlit-app/1.0 (+https://github.com/bukthorn/citydata)"

# ---- Overpass API: public instances, tried in order until one answers ----
# Current list: https://wiki.openstreetmap.org/wiki/Overpass_API#Public_Overpass_API_instances
OVERPASS_URLS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    # overpass.kumi.systems is left out: it often hangs for minutes before failing,
    # blocking one of the parallel slots; a pause and retry here is cheaper
]
# Server-side query limit; the client waits a bit longer so the server can report it
OVERPASS_QUERY_TIMEOUT_SEC = 240
OVERPASS_READ_TIMEOUT_SEC = 300
# After 429/504 the server asks to pause at least 30 s; repeated 429s get banned
OVERPASS_MIN_WAIT_SEC = 30.0
# A busy server spends ~35 s on any request, however small, so data is fetched
# in as few tiles as possible: start with tiles up to this size and only split
# a tile into 4 when the server rejects it as too heavy
BUILDING_TILE_KM = 10.0
BUILDING_MIN_TILE_KM = 2.5
# Tiles that can still be split get a shorter timeout and two rounds of tries
# (instant 502/429 glitches pass after the pause between rounds): splitting is
# cheaper than waiting for a heavy query to time out
SPLITTABLE_QUERY_TIMEOUT_SEC = 120
SPLITTABLE_ROUNDS = 2
# Requests at once: overpass-api.de allows 2 concurrent queries per IP,
# more get 429 and repeated 429s get banned
OVERPASS_PARALLEL = 2

# OSM data changes slowly: fetched results are shared by all visitors for a day
CACHE_TTL_SEC = 24 * 3600
# A cached 10 km tile of a dense city is ~1 MB (no geometry is kept);
# bounds memory use on Streamlit Cloud
TILE_CACHE_MAX_ENTRIES = 300


@st.cache_resource
def _shared_state() -> Dict[str, Any]:
    # One object for all reruns and visitors (module globals are reset on every
    # rerun). last_ok: mirror that answered last, tried first to skip busy ones.
    # heavy_tiles: tiles a server rejected as too heavy, fetched as quarters
    return {"last_ok": None, "heavy_tiles": set()}

# ---- Nominatim (geocoding): max 1 request/s per usage policy ----
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

AMENITIES = [
    "restaurant",
    "cafe",
    "bar",
    "pub",
    "fast_food",
    "food_court",
    "canteen",
]

RES_BUILDINGS = [
    "apartments",
    "residential",
    "house",
    "terrace",
    "semidetached_house",
    "detached",
    "dormitory",
    "bungalow",
]

DEFAULT_M2_PER_PERSON = 30.0
DEFAULT_FLOORS_BY_TYPE = {
    "bungalow": 1,
    "house": 2,
    "detached": 2,
    "semidetached_house": 2,
    "terrace": 2,
    "apartments": 4,
    "residential": 4,
    "dormitory": 5,
}

EARTH_RADIUS_M = 6371008.8


def _parse_levels(v: str) -> Optional[float]:
    # Values like "4.5", "3,5" (decimal comma), "2-3" (range) or "5;9" (sections
    # of different height): take the mean of all numbers found
    if v is None: return None
    nums = [float(n.replace(",", "."))
            for n in re.findall(r"\d+(?:[.,]\d+)?", str(v))]
    return sum(nums) / len(nums) if nums else None


def _parse_height_m(v: str) -> Optional[float]:
    if v is None: return None
    s = str(v).strip().lower()
    is_ft = "ft" in s
    num = ""
    dot_used = False
    for ch in s:
        if ch.isdigit(): num += ch
        elif ch == "." and not dot_used:
            num += ch
            dot_used = True
        elif num:
            break
    try:
        h = float(num) if num else None
    except:
        h = None
    if h is None: return None
    return h * 0.3048 if is_ft else h


def polygon_area_m2(geom: List[Dict[str, float]]) -> float:
    if not geom or len(geom) < 3: return 0.0
    lats = [p["lat"] for p in geom if "lat" in p and "lon" in p]
    lons = [p["lon"] for p in geom if "lat" in p and "lon" in p]
    if len(lats) < 3: return 0.0
    lat0 = sum(lats) / len(lats)
    lat0_rad = math.radians(lat0)
    xs, ys = [], []
    for lat, lon in zip(lats, lons):
        xs.append(EARTH_RADIUS_M * math.radians(lon) * math.cos(lat0_rad))
        ys.append(EARTH_RADIUS_M * math.radians(lat))
    area2 = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area2) * 0.5


def polygon_centroid_latlon(
        geom: List[Dict[str, float]]) -> Tuple[float, float]:
    lats = [p["lat"] for p in geom if "lat" in p and "lon" in p]
    lons = [p["lon"] for p in geom if "lat" in p and "lon" in p]
    if not lats: return 0.0, 0.0
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _join_rings(parts: List[List[Dict[str, float]]]) -> List[List[Dict[str, float]]]:
    # A multipolygon ring may be split into several ways: chain them by shared ends
    def key(p): return p.get("lat"), p.get("lon")
    pending = [list(p) for p in parts if len(p) >= 2]
    rings = []
    while pending:
        ring = pending.pop()
        while key(ring[0]) != key(ring[-1]):
            for i, p in enumerate(pending):
                if key(p[0]) == key(ring[-1]): ring += p[1:]
                elif key(p[-1]) == key(ring[-1]): ring += p[-2::-1]
                else: continue
                pending.pop(i)
                break
            else:
                break  # no continuation (broken relation): keep the ring as is
        rings.append(ring)
    return rings


def building_area_and_center(el: Dict[str, Any]) -> Tuple[float, float, float]:
    """Footprint area (m2) and center (lat, lon) of a way or multipolygon relation."""
    if el.get("type") == "relation":
        # Overpass "out geom" puts relation geometry into members, not "geometry"
        outer, inner = [], []
        for m in el.get("members", []):
            if m.get("type") != "way" or not m.get("geometry"): continue
            (inner if m.get("role") == "inner" else outer).append(m["geometry"])
        outer, inner = _join_rings(outer), _join_rings(inner)
    else:
        outer, inner = [el.get("geometry") or []], []
    # Inner rings are courtyards: subtract them
    area = (sum(polygon_area_m2(r) for r in outer) -
            sum(polygon_area_m2(r) for r in inner))
    c_lat, c_lon = polygon_centroid_latlon([p for r in outer for p in r])
    return area, c_lat, c_lon


def estimate_floors(building_type: str, tags: Dict[str, str]) -> float:
    lv = _parse_levels(tags.get("building:levels"))
    if lv is not None and lv > 0: return lv
    h = _parse_height_m(tags.get("height"))
    if h is not None and h > 0: return int(max(1, round(h / 3.0)))
    return int(DEFAULT_FLOORS_BY_TYPE.get(building_type, 2))


# Public servers are often all busy for a few minutes at once, so keep retrying
# with growing pauses (30, 60, 90 s) instead of failing fast
def overpass(query: str,
             rounds: int = 4,
             read_timeout: float = OVERPASS_READ_TIMEOUT_SEC) -> Dict[str, Any]:
    state = _shared_state()
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    errors = []
    for rnd in range(rounds):
        if rnd > 0:
            time.sleep(OVERPASS_MIN_WAIT_SEC * rnd)
        urls = OVERPASS_URLS
        last_ok = state["last_ok"]
        if last_ok in urls:
            urls = [last_ok] + [u for u in urls if u != last_ok]
        # Limits are per instance, so on 429/504 move on to the next mirror right away
        for url in urls:
            host = url.split("/")[2]
            try:
                r = requests.post(url,
                                  data=query.encode("utf-8"),
                                  headers=headers,
                                  timeout=(10, read_timeout))
            except requests.RequestException as e:
                errors.append(f"{host}: {type(e).__name__}")
                continue

            if r.status_code == 429:
                errors.append(f"{host}: HTTP 429 (too many requests)")
                continue
            if r.status_code == 504:
                errors.append(f"{host}: HTTP 504 (server busy)")
                continue
            if r.status_code == 400:
                # Query syntax error: other mirrors won't accept it either
                raise RuntimeError(
                    f"Overpass HTTP 400: {r.text[:300].strip()}")
            if r.status_code >= 400:
                # 406 etc. = this mirror blocks the client, others may not
                errors.append(f"{host}: HTTP {r.status_code}")
                continue

            try:
                data = r.json()
            except ValueError:
                errors.append(f"{host}: non-JSON response")
                continue
            # Server-side timeout/memory limit: HTTP 200 but the data is incomplete
            remark = data.get("remark") or ""
            if "runtime error" in remark.lower():
                errors.append(f"{host}: {remark.strip()}")
                continue
            state["last_ok"] = url
            return data

    raise RuntimeError("Overpass API did not respond. " + "; ".join(errors))


@st.cache_data(ttl=3600, show_spinner=False)
def get_city_center(city_name: str) -> Tuple[float, float]:
    r = requests.get(NOMINATIM_URL,
                     params={
                         "q": city_name,
                         "format": "jsonv2",
                         "featureType": "city",
                         "limit": 1,
                     },
                     headers={"User-Agent": USER_AGENT},
                     timeout=30)
    r.raise_for_status()
    results = r.json()
    if not results: raise ValueError(f"City '{city_name}' not found.")
    # Results are sorted by importance, so the first one is the best match
    best = results[0]
    return float(best["lat"]), float(best["lon"])


def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(
        dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# OSM data is fetched for minutes, so it is kept on disk (~/.streamlit/cache)
# Venues and buildings of a tile in one request: a busy server spends ~35 s on
# any request, so fewer requests is what makes loading faster. Each tile is
# cached as soon as it arrives, so after a failure the next run only fetches
# missing tiles, and another radius around the same center reuses them.
# Only a bbox filter: Overpass evaluates an added (around:) over the whole
# circle, making each tile as slow as the full query. Venues and buildings
# outside the circle are dropped later when they are assigned to rings
@st.cache_data(ttl=CACHE_TTL_SEC,
               max_entries=TILE_CACHE_MAX_ENTRIES,
               show_spinner=False)
def _fetch_tile(bbox, splittable):
    bb = ",".join(str(v) for v in bbox)
    b_re = "|".join(RES_BUILDINGS)
    amen_re = "|".join(AMENITIES)
    timeout = (SPLITTABLE_QUERY_TIMEOUT_SEC
               if splittable else OVERPASS_QUERY_TIMEOUT_SEC)
    q = f"""
    [out:json][timeout:{timeout}];
    (
      way({bb})["building"~"^({b_re})$"];
      relation({bb})["building"~"^({b_re})$"];
    );
    // qt: skip sorting by id, results are deduplicated anyway
    out tags geom qt;
    nwr({bb})["amenity"~"^({amen_re})$"];
    out center tags qt;
    """
    data = overpass(q,
                    rounds=SPLITTABLE_ROUNDS if splittable else 4,
                    read_timeout=timeout + 30)
    # Only what the analysis needs is kept: full building geometry would make
    # the cache tens of times bigger
    venues, buildings = [], []
    for el in data.get("elements", []):
        key = (el.get("type"), el.get("id"))
        tags = el.get("tags", {}) or {}
        # "out center" gives venues lat/lon (nodes) or a center (ways,
        # relations); "out geom" gives buildings geometry or members instead
        if "center" in el or el.get("type") == "node":
            c = el.get("center") or el
            if "lat" in c and "lon" in c:
                venues.append({
                    "key": key,
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "name": tags.get("name", ""),
                    "amenity": tags.get("amenity", "")
                })
        else:
            area_m2, c_lat, c_lon = building_area_and_center(el)
            if area_m2 <= 1.0: continue
            buildings.append({
                "key": key,
                "lat": c_lat,
                "lon": c_lon,
                "area_m2": area_m2,
                "floors": estimate_floors(tags.get("building", ""), tags)
            })
    return venues, buildings


def _tile_km(bbox):
    return (bbox[2] - bbox[0]) * math.radians(1.0) * EARTH_RADIUS_M / 1000


def _can_split(bbox):
    return _tile_km(bbox) / 2 >= BUILDING_MIN_TILE_KM


def _split_tile(bbox):
    s, w, n, e = bbox
    ms, mw = round((s + n) / 2, 6), round((w + e) / 2, 6)
    return [(s, w, ms, mw), (s, mw, ms, e), (ms, w, n, mw), (ms, mw, n, e)]


def _start_tiles(lat, lon, radius_m):
    # Tiles up to BUILDING_TILE_KM centered on the circle:
    # radius 5 km is one tile, 10 km is four quarters
    k = max(1, math.ceil(2 * radius_m / 1000 / BUILDING_TILE_KM))
    dlat = math.degrees(radius_m / EARTH_RADIUS_M)
    dlon = dlat / math.cos(math.radians(lat))
    tiles = []
    for i in range(k):
        for j in range(k):
            s = lat - dlat + 2 * dlat * i / k
            n = lat - dlat + 2 * dlat * (i + 1) / k
            w = lon - dlon + 2 * dlon * j / k
            e = lon - dlon + 2 * dlon * (j + 1) / k
            # Skip tiles that lie entirely outside the circle
            near_lat = min(max(lat, s), n)
            near_lon = min(max(lon, w), e)
            if haversine_m(lat, lon, near_lat, near_lon) > radius_m: continue
            tiles.append(tuple(round(v, 6) for v in (s, w, n, e)))
    return tiles


def fetch_city_data(lat, lon, radius_m, on_progress=None):
    """Venues (DataFrame) and residential buildings (list of dicts with lat,
    lon, area_m2, floors) in tiles covering the circle; on_progress gets 0-100."""
    def area(b): return (b[2] - b[0]) * (b[3] - b[1])

    tiles = _start_tiles(lat, lon, radius_m)
    total_area = sum(area(t) for t in tiles)
    done_area = 0.0
    venues, buildings, errors = {}, {}, []
    heavy_tiles = _shared_state()["heavy_tiles"]
    # Tiles go out in parallel, within the server's per-IP limit
    with ThreadPoolExecutor(max_workers=OVERPASS_PARALLEL) as pool:
        futures = {}

        def submit(bbox):
            if bbox in heavy_tiles and _can_split(bbox):
                for sub in _split_tile(bbox): submit(sub)
            else:
                futures[pool.submit(_fetch_tile, bbox, _can_split(bbox))] = bbox

        for t in tiles: submit(t)
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                bbox = futures.pop(fut)
                try:
                    tile_venues, tile_buildings = fut.result()
                except RuntimeError as e:
                    if _can_split(bbox):
                        # Too heavy for a busy server: retry as four quarters
                        heavy_tiles.add(bbox)
                        submit(bbox)
                        continue
                    # A failed tile doesn't stop the rest: they get cached
                    errors.append(e)
                else:
                    # Objects crossing a tile edge come back from both tiles
                    for v in tile_venues: venues[v["key"]] = v
                    for b in tile_buildings: buildings[b["key"]] = b
                done_area += area(bbox)
            if on_progress: on_progress(int(100 * done_area / total_area))
    if errors:
        raise RuntimeError(
            f"{len(errors)} tiles failed to load ({errors[0]}). Tiles loaded "
            "so far are cached, press the button again to load the rest.")
    df_pois = pd.DataFrame(list(venues.values()),
                           columns=["lat", "lon", "name", "amenity"])
    return df_pois, list(buildings.values())


def main():
    st.set_page_config(layout="wide", page_title="City POI Analysis")
    st.title("City data analysis")

    if "cities_data" not in st.session_state:
        st.session_state.cities_data = []

    with st.sidebar:
        st.header("Add City")
        city_input = st.text_input("City Name", "Moscow")
        max_rings_input = st.number_input("Radius (km)",
                                          min_value=1,
                                          value=10)
        if st.button("Calculate and Add"):
            progress = st.empty()
            try:
                with st.spinner(f"Loading data for {city_input}..."):
                    lat, lon = get_city_center(city_input)
                    radius_m = max_rings_input * 1000

                    progress.progress(0, text="Loading data: 0%")
                    df_pois, b_elements = fetch_city_data(
                        lat, lon, radius_m,
                        lambda pct: progress.progress(
                            pct, text=f"Loading data: {pct}%"))
                    progress.empty()
                    if df_pois.empty:
                        st.warning(f"No venues found in {city_input}.")
                    else:
                        df_pois["dist_km"] = df_pois.apply(
                            lambda r: haversine_m(lat, lon, r["lat"], r["lon"]
                                                  ) / 1000,
                            axis=1)
                        df_pois["ring"] = df_pois["dist_km"].astype(int)
                        df_pois = df_pois[df_pois["ring"] < max_rings_input]

                        rings = pd.DataFrame({"ring": range(max_rings_input)})
                        counts = df_pois.groupby("ring").size().reset_index(
                            name="count")
                        rings = rings.merge(counts, on="ring",
                                            how="left").fillna(0)
                        rings["count"] = rings["count"].astype(int)

                        rings["area_km2"] = rings["ring"].apply(
                            lambda r: math.pi * ((r + 1)**2 - r**2))
                        rings["density"] = rings["count"] / rings["area_km2"]

                        b_rows = []
                        for b in b_elements:
                            dist_m = haversine_m(lat, lon, b["lat"], b["lon"])
                            ring_k = int(dist_m // 1000)
                            if 0 <= ring_k < max_rings_input:
                                b_rows.append({
                                    "ring":
                                    ring_k,
                                    "floor_area_m2":
                                    b["area_m2"] * max(1, b["floors"])
                                })

                        if b_rows:
                            b_df = pd.DataFrame(b_rows).groupby(
                                "ring")["floor_area_m2"].sum().reset_index()
                            rings = rings.merge(b_df, on="ring",
                                                how="left").fillna(0)
                        else:
                            rings["floor_area_m2"] = 0

                        # People per ring from housing in that ring only; scaling
                        # to the official city population would pack the whole
                        # city into the circle when it is smaller than the city
                        rings["pop_est"] = rings[
                            "floor_area_m2"] / DEFAULT_M2_PER_PERSON

                        rings["pop_density"] = rings["pop_est"] / rings["area_km2"]

                        def _per1k(row):
                            return row["count"] / (
                                row["pop_est"] /
                                1000) if row["pop_est"] > 0 else 0

                        rings["per1k"] = rings.apply(_per1k, axis=1)

                        st.session_state.cities_data.append({
                            "city": city_input,
                            "center": {"lat": float(lat), "lon": float(lon)},
                            "pois": df_pois[["lat", "lon", "name", "amenity"]].copy(),
                            "rings": rings
                        })
                        st.success(f"City {city_input} added!")
            except Exception as e:
                progress.empty()
                st.error(f"Error: {e}")

        if st.button("Clear All Data"):
            st.session_state.cities_data = []
            st.rerun()

    if st.session_state.cities_data:
        st.subheader("Data Visualization")

        mode = st.radio("",
                        options=[
                            "Amount of cafes", "Population density",
                            "Cafes per 1k people", "Population (est)"
                        ],
                        horizontal=True,
                        label_visibility="collapsed")

        metric_map = {
            "Amount of cafes": "count",
            "Population density": "pop_density",
            "Cafes per 1k people": "per1k",
            "Population (est)": "pop_est"
        }
        y_col = metric_map[mode]

        # Use Plotly for interactive charts
        fig = go.Figure()
        for data in st.session_state.cities_data:
            # We must ensure data is passed as standard Python types to avoid "LargeUtf8" errors with Arrow
            fig.add_trace(
                go.Scatter(
                    x=[int(x + 1) for x in data["rings"]["ring"]],
                    y=[float(y) for y in data["rings"][y_col]],
                    mode='lines+markers',
                    name=str(data["city"]),
                    hovertemplate=
                    "<b>%{fullData.name}</b><br>Km: %{x}<br>Value: %{y:.2f}<extra></extra>"
                ))

        fig.update_layout(xaxis_title="Distance from center (km)",
                          yaxis_title=mode,
                          hovermode="x unified",
                          height=600,
                          template="plotly_white",
                          legend=dict(itemclick="toggle",
                                      itemdoubleclick="toggleothers"))

        # Display the chart using st.plotly_chart
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Map: POIs used for count")

        col_mm_label, col_mm_radio = st.columns([0.3, 3.7], vertical_alignment="center")
        with col_mm_label:
            st.markdown("**Map mode**")
        with col_mm_radio:
            map_mode = st.radio(
                "",
                options=["POI", "Heatmap"],
                index=0,
                horizontal=True,
                label_visibility="collapsed"
    )

        city_names = [d["city"] for d in st.session_state.cities_data]

        col_city_label, col_city_select = st.columns([0.3, 3.7], vertical_alignment="center")
        with col_city_label:
            st.markdown("**Map for city**")
        with col_city_select:
            selected_city = st.selectbox(
                "",
                city_names,
                index=len(city_names) - 1,
                label_visibility="collapsed"
            )

        city_data = next(d for d in st.session_state.cities_data if d["city"] == selected_city)
        pois_df = city_data["pois"]
        center = city_data["center"]

        if pois_df is None or pois_df.empty:
            st.info("No POIs to show on the map for this city.")
        
        else:
            # --- Map via Plotly Map (MapLibre): legend overlays map + interactive toggle by type ---
            pois_df = pois_df.copy()

            # гарантируем наличие amenity и приводим к строке
            if "amenity" not in pois_df.columns:
                pois_df["amenity"] = ""
            pois_df["poi_type"] = pois_df["amenity"].fillna("").astype(str)

            # Цветовая палитра (без чёрного/белого) — нужна только для режима POI
            color_map = {
                "restaurant": "rgb(231,76,60)",
                "cafe": "rgb(46,204,113)",
                "bar": "rgb(155,89,182)",
                "pub": "rgb(241,196,15)",
                "fast_food": "rgb(230,126,34)",
                "food_court": "rgb(52,152,219)",
                "canteen": "rgb(26,188,156)",
            }
            default_color = "rgb(149,165,166)"
            present_types = [t for t in color_map.keys() if (pois_df["poi_type"] == t).any()]

            map_fig = go.Figure()

            if map_mode == "POI":
                # --- POI mode (как сейчас): отдельный trace на каждый тип => легенда интерактивная ---
                for t in present_types:
                    sub = pois_df[pois_df["poi_type"] == t]
                    map_fig.add_trace(
                        go.Scattermap(
                            lat=sub["lat"].astype(float).tolist(),
                            lon=sub["lon"].astype(float).tolist(),
                            mode="markers",
                            name=t.replace("_", " "),
                            marker=dict(size=8, color=color_map.get(t, default_color)),
                            customdata=sub[["name", "poi_type"]].astype(str).values.tolist(),
                            hovertemplate="<b>%{customdata[0]}</b><br>Type: %{customdata[1]}<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>",
                        )
                    )

                # Остальные типы (если есть) — отдельная серия
                other = pois_df[~pois_df["poi_type"].isin(present_types)]
                if not other.empty:
                    map_fig.add_trace(
                        go.Scattermap(
                            lat=other["lat"].astype(float).tolist(),
                            lon=other["lon"].astype(float).tolist(),
                            mode="markers",
                            name="other",
                            marker=dict(size=8, color=default_color),
                            customdata=other[["name", "poi_type"]].astype(str).values.tolist(),
                            hovertemplate="<b>%{customdata[0]}</b><br>Type: %{customdata[1]}<br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>",
                        )
                    )

                map_fig.update_layout(
                    map=dict(
                        style="carto-darkmatter",
                        center=dict(lat=float(center["lat"]), lon=float(center["lon"])),
                        zoom=10,
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=520,
                    legend=dict(
                        x=0.01, y=0.99,
                        xanchor="left", yanchor="top",
                        bgcolor="rgba(30,30,30,0.9)",
                        bordercolor="rgba(0,0,0,0.12)",
                        borderwidth=1,
                        itemclick="toggle",
                        itemdoubleclick="toggleothers",
                    ),
                )

            else:
                # --- Heatmap mode: реальная метрика POI per 1 km² (квадраты 1×1 км) ---
                lat0 = float(center["lat"])
                lon0 = float(center["lon"])
                R = 6371008.8
                cos_lat0 = math.cos(math.radians(lat0))

                tmp = pois_df.copy()

                # x,y в метрах (приближённая локальная проекция относительно центра)
                tmp["x_m"] = R * math.radians(1.0) * (tmp["lon"].astype(float) - lon0) * cos_lat0
                tmp["y_m"] = R * math.radians(1.0) * (tmp["lat"].astype(float) - lat0)

                cell = 1000.0  # 1 км
                tmp["cx"] = (tmp["x_m"] // cell).astype(int)
                tmp["cy"] = (tmp["y_m"] // cell).astype(int)

                # Count POI в каждой ячейке 1×1 км (это и есть POI/км²)
                grid = tmp.groupby(["cx", "cy"], as_index=False).size().rename(columns={"size": "poi_per_km2"})
                grid["cell_id"] = grid["cx"].astype(str) + "_" + grid["cy"].astype(str)

                # --- Smoothing 3x3 (center=4, sides=2, diagonals=1) ---
                vals = {(int(r.cx), int(r.cy)): float(r.poi_per_km2) for r in grid.itertuples(index=False)}

                kernel = {
                    (0, 0): 4,
                    (1, 0): 2, (-1, 0): 2, (0, 1): 2, (0, -1): 2,
                    (1, 1): 1, (1, -1): 1, (-1, 1): 1, (-1, -1): 1,
                }
                w_sum = sum(kernel.values())

                smoothed = []
                for r in grid.itertuples(index=False):
                    cx = int(r.cx)
                    cy = int(r.cy)
                    acc = 0.0
                    for (dx, dy), w in kernel.items():
                        acc += w * vals.get((cx + dx, cy + dy), 0.0)
                    smoothed.append(acc / w_sum)

                grid["poi_per_km2_smoothed"] = smoothed

                # Функция перевода метрических смещений -> lat/lon
                def xy_to_latlon(x_m: float, y_m: float) -> tuple[float, float]:
                    lat = lat0 + (y_m / (R * math.radians(1.0)))
                    lon = lon0 + (x_m / (R * math.radians(1.0) * cos_lat0))
                    return lat, lon

                # Строим GeoJSON FeatureCollection из квадратов
                features = []
                for _, row in grid.iterrows():
                    cx = int(row["cx"])
                    cy = int(row["cy"])
                    cell_id = row["cell_id"]

                    x0 = cx * cell
                    x1 = (cx + 1) * cell
                    y0 = cy * cell
                    y1 = (cy + 1) * cell

                    # 4 угла квадрата (в lon/lat для GeoJSON)
                    lat_a, lon_a = xy_to_latlon(x0, y0)  # нижний-левый
                    lat_b, lon_b = xy_to_latlon(x1, y0)  # нижний-правый
                    lat_c, lon_c = xy_to_latlon(x1, y1)  # верхний-правый
                    lat_d, lon_d = xy_to_latlon(x0, y1)  # верхний-левый

                    features.append(
                        {
                            "type": "Feature",
                            "id": cell_id,
                            "properties": {"cell_id": cell_id},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [lon_a, lat_a],
                                    [lon_b, lat_b],
                                    [lon_c, lat_c],
                                    [lon_d, lat_d],
                                    [lon_a, lat_a],
                                ]],
                            },
                        }
                    )

                geojson = {"type": "FeatureCollection", "features": features}

                # Рисуем квадраты как choropleth
                map_fig.add_trace(
                    go.Choroplethmap(
                        geojson=geojson,
                        locations=grid["cell_id"].tolist(),
                        z=grid["poi_per_km2"].astype(float).tolist(),
                        featureidkey="id",
                        colorscale="Turbo",
                        zmin=0,
                        marker_opacity=0.75,
                        marker_line_width=0,  # без границ, чтобы выглядело как heatmap
                        colorbar=dict(
                            title="POI / 1 km²",
                            thickness=14,
                            len=0.55,
                        ),
                        hovertemplate="POI / 1 km²: <b>%{z}</b><extra></extra>",
                    )
                )

                map_fig.update_layout(
                    map=dict(
                        style="carto-darkmatter",
                        center=dict(lat=lat0, lon=lon0),
                        zoom=10,
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=520,
                    showlegend=False,
                )            

            st.plotly_chart(map_fig, use_container_width=True)

    else:
        st.info("Add a city in the sidebar to start analysis.")


if __name__ == "__main__":
    main()
