import math
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

# ---- Overpass mirrors (rotate on failure) ----
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]

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


def _parse_int_loose(v: str) -> Optional[int]:
    if v is None: return None
    s = "".join(ch for ch in str(v) if ch.isdigit())
    try:
        return int(s) if s else None
    except:
        return None


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


def estimate_floors(building_type: str, tags: Dict[str, str]) -> int:
    lv = _parse_int_loose(tags.get("building:levels"))
    if lv is not None and lv > 0: return lv
    h = _parse_height_m(tags.get("height"))
    if h is not None and h > 0: return int(max(1, round(h / 3.0)))
    return int(DEFAULT_FLOORS_BY_TYPE.get(building_type, 2))


def overpass(query: str,
             retries: int = 6,
             backoff_sec: float = 3.0) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    last_err = None
    for attempt in range(retries):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            r = requests.post(url,
                              data=query.encode("utf-8"),
                              headers=headers,
                              timeout=240)
            if r.status_code >= 400:
                raise requests.HTTPError(f"Overpass HTTP {r.status_code}")
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff_sec * (attempt + 1))
    raise last_err


def parse_population(tags: Dict[str, str]) -> int:
    if not tags: return -1
    val = tags.get("population") or tags.get("pop")
    if not val: return -1
    cleaned = "".join(ch for ch in str(val) if ch.isdigit())
    try:
        return int(cleaned) if cleaned else -1
    except:
        return -1


def get_city_center(city_name: str) -> Tuple[float, float, Dict[str, str]]:
    city_esc = city_name.replace('"', r'\"')
    q = f"""
    [out:json][timeout:180];
    (
      node["place"="city"]["name"~"^{city_esc}$",i];
      node["place"="city"]["name:en"~"^{city_esc}$",i];
      node["place"="city"]["int_name"~"^{city_esc}$",i];
      node["place"="city"]["official_name"~"^{city_esc}$",i];
      node["place"="city"]["alt_name"~"^{city_esc}$",i];
    );
    out body;
    """
    data = overpass(q)
    elements = data.get("elements", [])
    if not elements: raise ValueError(f"City '{city_name}' not found.")
    best = max(elements, key=lambda el: parse_population(el.get("tags", {})))
    return float(best["lat"]), float(best["lon"]), best.get("tags", {}) or {}


def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(
        dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2)**2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_pois(lat, lon, radius_m):
    amen_re = "|".join(AMENITIES)
    q = f"""
    [out:json][timeout:240];
    (
      nwr(around:{radius_m},{lat},{lon})["amenity"~"^({amen_re})$"];
    );
    out center tags;
    """
    data = overpass(q)
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        center = el.get("center") or el
        if "lat" in center and "lon" in center:
            rows.append({
                "lat": center["lat"],
                "lon": center["lon"],
                "name": tags.get("name", ""),
                "amenity": tags.get("amenity", "")
            })
    return pd.DataFrame(rows)


def fetch_residential(lat, lon, radius_m):
    b_re = "|".join(RES_BUILDINGS)
    q = f"""
    [out:json][timeout:240];
    (
      way(around:{radius_m},{lat},{lon})["building"~"^({b_re})$"];
      relation(around:{radius_m},{lat},{lon})["building"~"^({b_re})$"];
    );
    out tags geom;
    """
    data = overpass(q)
    return data.get("elements", [])


def main():
    st.set_page_config(layout="wide", page_title="City POI Analysis")
    st.title("City data analysis")

    if "cities_data" not in st.session_state:
        st.session_state.cities_data = []

    with st.sidebar:
        st.header("Add City")
        city_input = st.text_input("City Name", "Moscow")
        max_rings_input = st.number_input("Distance (km)",
                                          min_value=1,
                                          value=10)
        if st.button("Calculate and Add"):
            try:
                with st.spinner(f"Loading data for {city_input}..."):
                    lat, lon, center_tags = get_city_center(city_input)
                    radius_m = max_rings_input * 1000

                    df_pois = fetch_pois(lat, lon, radius_m)
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

                        b_elements = fetch_residential(lat, lon, radius_m)
                        b_rows = []
                        for el in b_elements:
                            tags = el.get("tags", {}) or {}
                            geom = el.get("geometry")
                            if not geom or len(geom) < 3: continue
                            area_m2 = polygon_area_m2(geom)
                            if area_m2 <= 1.0: continue
                            c_lat, c_lon = polygon_centroid_latlon(geom)
                            dist_m = haversine_m(lat, lon, c_lat, c_lon)
                            ring_k = int(dist_m // 1000)
                            if 0 <= ring_k < max_rings_input:
                                floors = estimate_floors(
                                    tags.get("building", ""), tags)
                                b_rows.append({
                                    "ring":
                                    ring_k,
                                    "floor_area_m2":
                                    area_m2 * max(1, floors)
                                })

                        if b_rows:
                            b_df = pd.DataFrame(b_rows).groupby(
                                "ring")["floor_area_m2"].sum().reset_index()
                            rings = rings.merge(b_df, on="ring",
                                                how="left").fillna(0)
                        else:
                            rings["floor_area_m2"] = 0

                        city_pop = parse_population(center_tags)
                        total_floor_area = rings["floor_area_m2"].sum()
                        if city_pop > 0 and total_floor_area > 0:
                            rings["pop_est"] = rings["floor_area_m2"] * (
                                city_pop / total_floor_area)
                        else:
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

        # Выбор города для карты (если городов несколько)
        city_names = [d["city"] for d in st.session_state.cities_data]
        selected_city = st.selectbox("Map for city", city_names, index=len(city_names) - 1)

        city_data = next(d for d in st.session_state.cities_data if d["city"] == selected_city)
        pois_df = city_data["pois"]
        center = city_data["center"]

        if pois_df is None or pois_df.empty:
            st.info("No POIs to show on the map for this city.")
        else:
            # PyDeck map (Mapbox-style). Если нет MAPBOX_TOKEN, всё равно отрисуется базовая карта.
            view_state = pdk.ViewState(
                latitude=float(center["lat"]),
                longitude=float(center["lon"]),
                zoom=10,
                pitch=0
            )

            # ---- Тип POI для раскраски: только amenity (shop=coffee удалён из проекта)
            pois_df = pois_df.copy()

            # гарантируем наличие amenity и приводим к строке
            if "amenity" not in pois_df.columns:
                pois_df["amenity"] = ""

            pois_df["poi_type"] = pois_df["amenity"].fillna("").astype(str)

            # ---- Цветовая палитра (без чёрного/белого)
            color_map = {
                "restaurant": [231, 76, 60],     # красный
                "cafe": [46, 204, 113],          # зелёный
                "bar": [155, 89, 182],           # фиолетовый
                "pub": [241, 196, 15],           # жёлтый
                "fast_food": [230, 126, 34],     # оранжевый
                "food_court": [52, 152, 219],    # синий
                "canteen": [26, 188, 156],       # бирюзовый
            }
            default_color = [149, 165, 166]      # серо-голубой (не белый/чёрный)

            pois_df["color"] = pois_df["poi_type"].apply(lambda t: color_map.get(t, default_color))

            # ---- Легенда (вверху справа над картой, прижата вправо)
            legend_items = []
            present_types = [t for t in color_map.keys() if (pois_df["poi_type"] == t).any()]
            for t in present_types:
                rgb = color_map[t]
                label = t.replace("_", " ")
                legend_items.append(
                    f"""
                    <div style="display:flex;align-items:center;gap:8px;margin:2px 0;">
                    <div style="width:12px;height:12px;border-radius:50%;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});"></div>
                    <div style="font-size:12px;line-height:12px;">{label}</div>
                    </div>
                    """
                )

            if legend_items:
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:flex-end;">
                    <div style="
                        background: rgba(50,50,50,0.92);
                        padding:10px 12px;
                        border-radius:10px;
                        border: 1px solid rgba(0,0,0,0.08);
                        max-width: 260px;
                    ">
                        <div style="font-weight:600;font-size:13px;margin-bottom:6px;">POI legend</div>
                        {''.join(legend_items)}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ---- PyDeck слой точек
            view_state = pdk.ViewState(
                latitude=float(center["lat"]),
                longitude=float(center["lon"]),
                zoom=10,
                pitch=0
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=pois_df,
                get_position="[lon, lat]",
                get_radius=20,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
                stroked=False,
                filled=True,
            )

            tooltip = {
                "html": "<b>{name}</b><br/>Type: {poi_type}<br/>Lat: {lat}<br/>Lon: {lon}",
                "style": {"backgroundColor": "white", "color": "black"}
            }

            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            )

            st.pydeck_chart(deck, use_container_width=True)


    else:
        st.info("Add a city in the sidebar to start analysis.")


if __name__ == "__main__":
    main()
