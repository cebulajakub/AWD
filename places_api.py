import requests
import time
import pandas as pd


class PlacesDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://places.googleapis.com/v1/places:searchText"
        self.headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": (
                "places.displayName.text,"
                "places.rating,"
                "places.userRatingCount,"
                "places.priceLevel,"
                "places.primaryType,"
                "places.location,"
                "nextPageToken"
            ),
        }

    def fetch_places(self, query, max_pages=3):
        """Pobiera miejsca z API Google obsługując paginację."""
        all_places = []
        page_token = ""

        for page in range(max_pages):
            print(f"Pobieranie strony {page + 1}...")

            payload = {
                "textQuery": query,
                "languageCode": "pl",
                "maxResultCount": 20,
            }

            if page_token:
                payload["pageToken"] = page_token

            response = requests.post(self.url, json=payload, headers=self.headers)

            if response.status_code == 200:
                data = response.json()

                if "places" in data:
                    all_places.extend(data["places"])

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

                # Google wymaga krótkiej przerwy przed użyciem nextPageToken
                time.sleep(2)
            else:
                print(f"Błąd HTTP {response.status_code}: {response.text}")
                break

        return all_places

    def process_and_clean_data(self, raw_places):
        """Przetwarza surowy JSON na Pandas DataFrame z dodatkowymi cechami."""
        processed_data = []

        for place in raw_places:
            name = place.get("displayName", {}).get("text", "Nieznane miejsce")
            rating = place.get("rating", 0.0)
            reviews_count = place.get("userRatingCount", 0)
            price_level = place.get("priceLevel", "PRICE_LEVEL_UNSPECIFIED")
            place_type = place.get("primaryType", "tourist_attraction")

            # Pobranie współrzędnych
            location = place.get("location", {})
            latitude = location.get("latitude")
            longitude = location.get("longitude")

            processed_data.append(
                {
                    "Nazwa": name,
                    "Kategoria_API": place_type,
                    "Ocena": rating,
                    "Liczba_Opinii": reviews_count,
                    "Poziom_Cen_API": price_level,
                    "Latitude": latitude,
                    "Longitude": longitude,
                }
            )

        df = pd.DataFrame(processed_data)

        if df.empty:
            return df

        # 1. Mapowanie kosztów
        price_mapping = {
            "PRICE_LEVEL_FREE": 0,
            "PRICE_LEVEL_INEXPENSIVE": 20,
            "PRICE_LEVEL_MODERATE": 50,
            "PRICE_LEVEL_EXPENSIVE": 120,
            "PRICE_LEVEL_VERY_EXPENSIVE": 250,
            "PRICE_LEVEL_UNSPECIFIED": 30,
        }
        df["Szacowany_Koszt_PLN"] = df["Poziom_Cen_API"].map(price_mapping).fillna(30)

        # 2. Szacowanie czasu zwiedzania
        time_mapping = {
            "museum": 120,
            "park": 60,
            "restaurant": 90,
            "cafe": 45,
            "tourist_attraction": 60,
            "art_gallery": 90,
            "church": 30,
            "zoo": 180,
            "amusement_park": 240,
        }
        df["Szacowany_Czas_MIN"] = df["Kategoria_API"].map(time_mapping).fillna(60)

        # Odrzucamy miejsca bez ocen
        df = df[df["Liczba_Opinii"] > 50].reset_index(drop=True)

        return df


# --- UŻYCIE KLASY ---
if __name__ == "__main__":
    API_KEY = "TUTAJ_WKLEJ_SWÓJ_KLUCZ_API"

    collector = PlacesDataCollector(API_KEY)

    print("Rozpoczynam pobieranie danych...")
    surowe_dane = collector.fetch_places(
        "najlepsze atrakcje turystyczne i muzea we Wrocławiu",
        max_pages=3
    )

    print("Przetwarzanie danych i inżynieria cech...")
    df_atrakcje = collector.process_and_clean_data(surowe_dane)

    if df_atrakcje.empty:
        print("Brak danych do zapisania.")
    else:
        print("\nPodgląd przygotowanych danych:")
        print(
            df_atrakcje[
                [
                    "Nazwa",
                    "Kategoria_API",
                    "Latitude",
                    "Longitude",
                    "Szacowany_Czas_MIN",
                    "Szacowany_Koszt_PLN",
                ]
            ].head()
        )

        df_atrakcje.to_csv(
            "baza_atrakcji_wroclaw.csv",
            index=False,
            encoding="utf-8"
        )
        print("\nDane zapisano pomyślnie do pliku: baza_atrakcji_wroclaw.csv")