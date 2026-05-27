import math
import requests
import streamlit as st
import pandas as pd
import pulp
import folium
from streamlit_folium import st_folium

# ==========================================
# KONFIGURACJA
# ==========================================
st.set_page_config(page_title="Asystent Turysty - AWD", page_icon="🌍", layout="wide")
st.title("🌍 Inteligentny Asystent Turystyczny (AWD)")
st.markdown(
    "System wspomagania decyzji optymalizujący wybór atrakcji oraz kolejność ich odwiedzania "
    "z wykorzystaniem realistycznych tras po sieci dróg (OSRM)."
)

OSRM_BASE_URL = "https://router.project-osrm.org"

# Stan sesji, żeby wynik nie znikał po rerunach
if "plan_data" not in st.session_state:
    st.session_state.plan_data = None


# ==========================================
# WCZYTYWANIE DANYCH
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("baza_atrakcji_wroclaw.csv")

        required_columns = [
            "Nazwa",
            "Kategoria_API",
            "Ocena",
            "Liczba_Opinii",
            "Poziom_Cen_API",
            "Latitude",
            "Longitude",
            "Szacowany_Koszt_PLN",
            "Szacowany_Czas_MIN",
        ]

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error(f"Brak wymaganych kolumn w CSV: {missing}")
            return None

        df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
        return df

    except FileNotFoundError:
        st.error("Błąd: Nie znaleziono pliku baza_atrakcji_wroclaw.csv!")
        return None


# ==========================================
# FUNKCJE OSRM
# ==========================================
def _coords_to_osrm_string(points):
    return ";".join(f"{p['lon']},{p['lat']}" for p in points)


@st.cache_data(show_spinner=False)
def get_osrm_table(points_tuple, profile):
    points = [{"name": p[0], "lat": p[1], "lon": p[2]} for p in points_tuple]
    coords = _coords_to_osrm_string(points)

    url = f"{OSRM_BASE_URL}/table/v1/{profile}/{coords}?annotations=duration,distance"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("code") != "Ok":
        raise ValueError(f"OSRM Table error: {data}")

    durations = data.get("durations")
    distances = data.get("distances")

    if durations is None or distances is None:
        raise ValueError("OSRM nie zwrócił durations/distances.")

    return durations, distances


@st.cache_data(show_spinner=False)
def get_osrm_route_geometry(points_tuple, route_tuple, profile):
    points = [{"name": p[0], "lat": p[1], "lon": p[2]} for p in points_tuple]
    ordered_points = [points[idx] for idx in route_tuple]
    coords = _coords_to_osrm_string(ordered_points)

    url = (
        f"{OSRM_BASE_URL}/route/v1/{profile}/{coords}"
        "?overview=full&geometries=geojson&steps=false"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("code") != "Ok":
        raise ValueError(f"OSRM Route error: {data}")

    routes = data.get("routes", [])
    if not routes:
        raise ValueError("OSRM nie zwrócił żadnej trasy.")

    route = routes[0]
    geometry = route["geometry"]["coordinates"]  # [lon, lat]
    distance_m = route["distance"]
    duration_s = route["duration"]

    return geometry, distance_m, duration_s


# ==========================================
# FUNKCJE TSP / POMOCNICZE
# ==========================================
def build_points_tuple(points):
    return tuple((p["name"], float(p["lat"]), float(p["lon"])) for p in points)


def nearest_neighbor_fallback_from_matrix(duration_matrix):
    n = len(duration_matrix)
    if n <= 1:
        return [0]

    unvisited = set(range(1, n))
    route = [0]
    current = 0

    while unvisited:
        reachable = [j for j in unvisited if duration_matrix[current][j] is not None]
        if not reachable:
            route.extend(sorted(list(unvisited)))
            break

        next_node = min(reachable, key=lambda j: duration_matrix[current][j])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return route


def solve_tsp_path_from_duration_matrix(duration_matrix):
    n = len(duration_matrix)

    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]

    BIG_M = 10**7

    cost = [[BIG_M for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if duration_matrix[i][j] is not None:
                cost[i][j] = float(duration_matrix[i][j])

    prob = pulp.LpProblem("TSP_Path_OSRM", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x",
        ((i, j) for i in range(n) for j in range(n) if i != j),
        cat="Binary",
    )

    u = pulp.LpVariable.dicts(
        "u",
        range(n),
        lowBound=0,
        upBound=n - 1,
        cat="Continuous",
    )

    y = pulp.LpVariable.dicts("y", range(1, n), cat="Binary")

    prob += pulp.lpSum(
        cost[i][j] * x[(i, j)]
        for i in range(n)
        for j in range(n)
        if i != j
    )

    prob += pulp.lpSum(x[(0, j)] for j in range(1, n)) == 1
    prob += pulp.lpSum(x[(j, 0)] for j in range(1, n)) == 0

    for j in range(1, n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    for i in range(1, n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) + y[i] == 1

    prob += pulp.lpSum(y[i] for i in range(1, n)) == 1

    prob += u[0] == 0
    for i in range(1, n):
        prob += u[i] >= 1
        prob += u[i] <= n - 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n - 1) * x[(i, j)] <= n - 2

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    route = [0]
    current = 0
    visited = {0}

    while len(route) < n:
        next_node = None
        for j in range(n):
            if current != j and (current, j) in x and pulp.value(x[(current, j)]) > 0.5:
                next_node = j
                break

        if next_node is None or next_node in visited:
            break

        route.append(next_node)
        visited.add(next_node)
        current = next_node

    if len(route) != n:
        return None

    return route


def compute_route_summary_from_matrix(points, route, duration_matrix, distance_matrix):
    total_distance_km = 0.0
    total_travel_minutes = 0.0
    route_segments = []

    for idx in range(len(route) - 1):
        a_idx = route[idx]
        b_idx = route[idx + 1]

        a = points[a_idx]
        b = points[b_idx]

        dist_m = distance_matrix[a_idx][b_idx]
        dur_s = duration_matrix[a_idx][b_idx]

        if dist_m is None or dur_s is None:
            dist_km = 0.0
            travel_min = 0.0
        else:
            dist_km = dist_m / 1000.0
            travel_min = dur_s / 60.0

        total_distance_km += dist_km
        total_travel_minutes += travel_min

        route_segments.append(
            {
                "from": a["name"],
                "to": b["name"],
                "distance_km": dist_km,
                "travel_min": travel_min,
            }
        )

    return total_distance_km, total_travel_minutes, route_segments


def create_map(points, route, geometry_lonlat):
    center_lat = sum(p["lat"] for p in points) / len(points)
    center_lon = sum(p["lon"] for p in points) / len(points)

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for idx, point in enumerate(points):
        if idx == 0:
            marker_label = "START"
            folium.Marker(
                [point["lat"], point["lon"]],
                popup=f"<b>START</b><br>{point['name']}",
                tooltip=f"START: {point['name']}",
                icon=folium.Icon(color="green", icon="play"),
            ).add_to(fmap)
        else:
            order_pos = route.index(idx) if idx in route else "?"
            marker_label = str(order_pos)
            folium.Marker(
                [point["lat"], point["lon"]],
                popup=(
                    f"<b>{order_pos}. {point['name']}</b><br>"
                    f"Kategoria: {point.get('category', 'brak')}<br>"
                    f"Ocena: {point.get('rating', 'brak')}"
                ),
                tooltip=f"{order_pos}. {point['name']}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(fmap)

        folium.map.Marker(
            [point["lat"], point["lon"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size: 12px;
                    font-weight: bold;
                    color: black;
                    background: white;
                    border: 1px solid gray;
                    border-radius: 8px;
                    padding: 2px 6px;
                    white-space: nowrap;
                ">
                    {marker_label}
                </div>
                """
            ),
        ).add_to(fmap)

    line_points = [[lat, lon] for lon, lat in geometry_lonlat]
    folium.PolyLine(line_points, weight=5, opacity=0.85).add_to(fmap)

    return fmap


def oblicz_uzytecznosc(row, preferencje):
    kategoria = row["Kategoria_API"]
    waga = preferencje.get(kategoria, 3)
    ocena = row["Ocena"] if row["Ocena"] > 0 else 3.0
    liczba_opinii = row["Liczba_Opinii"] if row["Liczba_Opinii"] > 0 else 1
    bonus_popularnosci = math.log1p(liczba_opinii)
    return waga * ocena + 0.3 * bonus_popularnosci


# ==========================================
# GŁÓWNY PROGRAM
# ==========================================
df = load_data()

if df is not None:
    st.sidebar.header("⚙️ Twoje ograniczenia")

    budzet = st.sidebar.slider(
        "Maksymalny budżet (PLN)", min_value=0, max_value=500, value=150, step=10
    )
    czas_godziny = st.sidebar.slider(
        "Dostępny czas (godziny)", min_value=1.0, max_value=12.0, value=5.0, step=0.5
    )
    czas_minuty = int(czas_godziny * 60)

    st.sidebar.markdown("---")
    st.sidebar.header("📍 Punkt startowy")
    start_lat = st.sidebar.number_input(
        "Szerokość geograficzna startu",
        value=51.1090,
        format="%.6f",
    )
    start_lon = st.sidebar.number_input(
        "Długość geograficzna startu",
        value=17.0327,
        format="%.6f",
    )
    start_name = st.sidebar.text_input(
        "Nazwa punktu startowego",
        value="Rynek we Wrocławiu",
    )

    # Zawsze samochód
    osrm_profile = "driving"
    transport_mode = "samochód"

    st.sidebar.markdown("---")
    st.sidebar.header("❤️ Twoje preferencje (1-5)")
    st.sidebar.caption("1 = omijam, 5 = uwielbiam")

    pref_muzea = st.sidebar.slider("Muzea i Sztuka", 1, 5, 3)
    pref_parki = st.sidebar.slider("Parki i Natura", 1, 5, 3)
    pref_restauracje = st.sidebar.slider("Restauracje i Kawiarnie", 1, 5, 3)
    pref_zabytki = st.sidebar.slider("Zabytki i Architektura", 1, 5, 3)
    pref_inne = st.sidebar.slider("Inne atrakcje", 1, 5, 3)

    preferencje_turysty = {
        "museum": pref_muzea,
        "art_gallery": pref_muzea,
        "park": pref_parki,
        "zoo": pref_parki,
        "restaurant": pref_restauracje,
        "cafe": pref_restauracje,
        "church": pref_zabytki,
        "tourist_attraction": pref_inne,
        "amusement_park": pref_inne,
    }

    st.sidebar.markdown("---")
    uwzglednij_dojazdy = st.sidebar.checkbox(
        "Uwzględnij czas przemieszczania w podsumowaniu",
        value=True,
    )

    limit_atrakcji = st.sidebar.slider(
        "Maksymalna liczba atrakcji do zaplanowania",
        min_value=2,
        max_value=10,
        value=6,
        step=1,
    )

    if st.sidebar.button("🗑️ Wyczyść plan"):
        st.session_state.plan_data = None
        st.rerun()

    if st.button("🚀 Generuj optymalny plan wycieczki", type="primary"):
        with st.spinner("Algorytm przelicza optymalne warianty..."):
            df_opt = df.copy()
            df_opt["Utility_Score"] = df_opt.apply(
                lambda row: oblicz_uzytecznosc(row, preferencje_turysty),
                axis=1,
            )

            prob = pulp.LpProblem("Optymalizacja_Wycieczki", pulp.LpMaximize)
            indeksy_miejsc = list(df_opt.index)
            x = pulp.LpVariable.dicts("miejsce", indeksy_miejsc, cat="Binary")

            prob += pulp.lpSum(df_opt.loc[i, "Utility_Score"] * x[i] for i in indeksy_miejsc)

            prob += pulp.lpSum(
                df_opt.loc[i, "Szacowany_Koszt_PLN"] * x[i]
                for i in indeksy_miejsc
            ) <= budzet

            prob += pulp.lpSum(
                df_opt.loc[i, "Szacowany_Czas_MIN"] * x[i]
                for i in indeksy_miejsc
            ) <= czas_minuty

            prob += pulp.lpSum(x[i] for i in indeksy_miejsc) <= limit_atrakcji

            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            if pulp.LpStatus[prob.status] != "Optimal":
                st.error("Nie udało się znaleźć rozwiązania dla podanych ograniczeń.")
            else:
                wybrane_indeksy = [i for i in indeksy_miejsc if x[i].varValue == 1.0]

                if not wybrane_indeksy:
                    st.warning(
                        "Niestety, budżet lub czas są zbyt małe, aby odwiedzić jakąkolwiek atrakcję z bazy."
                    )
                else:
                    wynik_df = df_opt.loc[wybrane_indeksy].copy().reset_index(drop=True)

                    points = [
                        {
                            "name": start_name,
                            "lat": float(start_lat),
                            "lon": float(start_lon),
                            "category": "start",
                            "rating": "-",
                        }
                    ]

                    for _, row in wynik_df.iterrows():
                        points.append(
                            {
                                "name": row["Nazwa"],
                                "lat": float(row["Latitude"]),
                                "lon": float(row["Longitude"]),
                                "category": row["Kategoria_API"],
                                "rating": row["Ocena"],
                            }
                        )

                    points_tuple = build_points_tuple(points)

                    try:
                        duration_matrix, distance_matrix = get_osrm_table(points_tuple, osrm_profile)
                    except Exception as e:
                        st.error(f"Nie udało się pobrać macierzy tras z OSRM: {e}")
                        duration_matrix, distance_matrix = None, None

                    if duration_matrix is not None and distance_matrix is not None:
                        route = solve_tsp_path_from_duration_matrix(duration_matrix)
                        tsp_method = "Dokładny model TSP na realnych czasach OSRM (PuLP)"

                        if route is None:
                            route = nearest_neighbor_fallback_from_matrix(duration_matrix)
                            tsp_method = "Heurystyka najbliższego sąsiada na realnych czasach OSRM"

                        total_distance_km, total_travel_minutes, route_segments = compute_route_summary_from_matrix(
                            points,
                            route,
                            duration_matrix,
                            distance_matrix,
                        )

                        try:
                            geometry_lonlat, route_distance_m, route_duration_s = get_osrm_route_geometry(
                                points_tuple,
                                tuple(route),
                                osrm_profile,
                            )
                            total_distance_km = route_distance_m / 1000.0
                            total_travel_minutes = route_duration_s / 60.0
                        except Exception as e:
                            st.warning(
                                f"Nie udało się pobrać pełnej geometrii trasy z OSRM. "
                                f"Mapa pokaże tylko punkty. Szczegóły: {e}"
                            )
                            geometry_lonlat = [[points[idx]["lon"], points[idx]["lat"]] for idx in route]

                        suma_kosztow = wynik_df["Szacowany_Koszt_PLN"].sum()
                        suma_czasu_zwiedzania = wynik_df["Szacowany_Czas_MIN"].sum()
                        calkowita_uzytecznosc = wynik_df["Utility_Score"].sum()

                        if uwzglednij_dojazdy:
                            suma_czasu_calkowita = suma_czasu_zwiedzania + total_travel_minutes
                        else:
                            suma_czasu_calkowita = suma_czasu_zwiedzania

                        ordered_names = [points[idx]["name"] for idx in route[1:]]
                        wynik_df["Kolejność"] = wynik_df["Nazwa"].apply(
                            lambda name: ordered_names.index(name) + 1 if name in ordered_names else 999
                        )
                        wynik_df = wynik_df.sort_values("Kolejność").reset_index(drop=True)

                        st.session_state.plan_data = {
                            "wynik_df": wynik_df,
                            "route_segments": route_segments,
                            "points": points,
                            "route": route,
                            "geometry_lonlat": geometry_lonlat,
                            "suma_kosztow": float(suma_kosztow),
                            "suma_czasu_zwiedzania": float(suma_czasu_zwiedzania),
                            "total_travel_minutes": float(total_travel_minutes),
                            "total_distance_km": float(total_distance_km),
                            "suma_czasu_calkowita": float(suma_czasu_calkowita),
                            "calkowita_uzytecznosc": float(calkowita_uzytecznosc),
                            "tsp_method": tsp_method,
                            "czas_minuty": int(czas_minuty),
                        }

    # ==========================================
    # WIDOK WYNIKÓW Z SESSION STATE
    # ==========================================
    if st.session_state.plan_data is not None:
        plan = st.session_state.plan_data

        wynik_df = plan["wynik_df"]
        route_segments = plan["route_segments"]
        points = plan["points"]
        route = plan["route"]
        geometry_lonlat = plan["geometry_lonlat"]
        suma_kosztow = plan["suma_kosztow"]
        suma_czasu_zwiedzania = plan["suma_czasu_zwiedzania"]
        total_travel_minutes = plan["total_travel_minutes"]
        total_distance_km = plan["total_distance_km"]
        suma_czasu_calkowita = plan["suma_czasu_calkowita"]
        calkowita_uzytecznosc = plan["calkowita_uzytecznosc"]
        tsp_method = plan["tsp_method"]
        czas_minuty = plan["czas_minuty"]

        st.success("✅ Znaleziono optymalny plan wycieczki oraz realistyczną trasę!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Wykorzystany Budżet", f"{suma_kosztow:.0f} PLN")
        col2.metric("Czas zwiedzania", f"{suma_czasu_zwiedzania / 60:.1f} h")
        col3.metric("Czas przemieszczania", f"{total_travel_minutes / 60:.1f} h")
        col4.metric("Dystans trasy", f"{total_distance_km:.2f} km")

        col5, col6, col7 = st.columns(3)
        col5.metric("Łączny czas", f"{suma_czasu_calkowita / 60:.1f} h")
        col6.metric("Wynik Zadowolenia", f"{calkowita_uzytecznosc:.1f} pkt")
        col7.metric("Liczba atrakcji", f"{len(wynik_df)}")

        st.info(f"Metoda wyznaczenia kolejności: **{tsp_method}**")

        st.subheader("📍 Twój plan podróży")
        tabela_wyswietlana = wynik_df[
            [
                "Kolejność",
                "Nazwa",
                "Kategoria_API",
                "Szacowany_Czas_MIN",
                "Szacowany_Koszt_PLN",
                "Ocena",
                "Latitude",
                "Longitude",
            ]
        ].copy()

        tabela_wyswietlana.columns = [
            "Kolejność",
            "Nazwa Atrakcji",
            "Kategoria",
            "Czas zwiedzania (min)",
            "Koszt (PLN)",
            "Ocena Google",
            "Latitude",
            "Longitude",
        ]
        st.dataframe(tabela_wyswietlana, use_container_width=True)

        st.subheader("🧭 Szczegóły trasy")
        route_df = pd.DataFrame(route_segments)
        if not route_df.empty:
            route_df["distance_km"] = route_df["distance_km"].round(2)
            route_df["travel_min"] = route_df["travel_min"].round(1)
            route_df.columns = [
                "Skąd",
                "Dokąd",
                "Dystans (km)",
                "Czas przejazdu (min)",
            ]
            st.dataframe(route_df, use_container_width=True)

        st.subheader("🗺️ Wizualizacja trasy")
        fmap = create_map(points, route, geometry_lonlat)
        st_folium(fmap, width=None, height=650)

        if suma_czasu_calkowita > czas_minuty:
            st.warning(
                "Uwaga: po doliczeniu realnego czasu przemieszczania łączny czas "
                "przekracza limit użytkownika. Obecny model wyboru atrakcji nadal "
                "ogranicza tylko czas zwiedzania atrakcji."
            )

        st.caption(
            "Trasa, dystans i czas przemieszczania są pobierane z OSRM na podstawie "
            "rzeczywistej sieci dróg, a nie z odległości w linii prostej."
        )