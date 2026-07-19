# face-identity-clusterer

Grupowanie twarzy bez etykiet (unsupervised face clustering): **detekcja → embedding
(ArcFace 512D) → filtr jakości → tracking (wideo) → klasteryzacja (HDBSCAN) →
trwałe prototypy osób z re-identyfikacją między uruchomieniami**.

Implementacja koncepcji z `Wprowadzenie.md`.

## Instalacja

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/mac: source venv/bin/activate)
pip install -r requirements.txt
```

Przy pierwszym uruchomieniu InsightFace automatycznie pobierze paczkę modeli
`buffalo_l` (RetinaFace + ArcFace, ONNX, ~280 MB) do `~/.insightface/models/`.

GPU NVIDIA: zamień `onnxruntime` na `onnxruntime-gpu` i ustaw w `config.yaml`
`providers: [CUDAExecutionProvider, CPUExecutionProvider]`.

## Dwa etapy: zbieranie i analiza

Praca dzieli się na dwa zadania o przeciwnych profilach obciążenia, dlatego są
rozdzielone w czasie, a nie uruchamiane równolegle:

| Etap | Godziny | Co robi | Czy wymaga modeli |
|---|---|---|---|
| 1. Akwizycja | 07:00–15:00 | zbiera klatki z kamery do `dataset/` | **nie** |
| 2. Analiza | 15:00–07:00 | detekcja, klasteryzacja, trening klasyfikatora | tak |

```bash
# całodobowy nadzorca (blokuje; Ctrl+C przerywa)
python main.py schedule

# co zrobiłby teraz, bez uruchamiania
python main.py schedule --status

# jednorazowe wymuszenie analizy nocnej
python main.py schedule --analyze-now
```

Etap 1 nie potrzebuje InsightFace — próbkowanie czasowe, filtr ostrości i
detekcja zmian działają na samym OpenCV. Można więc zacząć zbierać materiał
zanim modele w ogóle zostaną pobrane.

### Samo zbieranie (bez harmonogramu)

```bash
python main.py capture --camera 0 --interval 1.0 --label "wejscie"
python main.py capture --camera 0 --max-frames 200 --display
python main.py capture --video ./nagranie.mp4          # także z pliku
python main.py dataset                                  # co już zebrano
```

Każda sesja to folder z klatkami JPEG i plikiem `manifest.json` (źródło,
etykieta, ustawienia, statystyki odrzuceń). Zwykły folder ze zdjęciami — bez
kroku importu, do obejrzenia w dowolnej przeglądarce plików.

### Trzy filtry przy zbieraniu

1. **interwał** — jedna klatka na `interval_seconds` czasu strumienia;
2. **ostrość** — wariancja Laplace'a, ta sama miara, której używa późniejszy
   filtr jakości;
3. **zmiana** — jaki **procent kadru** różni się od ostatniej zapisanej klatki.

Punkt 3 celowo mierzy *jaka część* obrazu się zmieniła, a nie *o ile średnio*
się zmieniła. Uśrednianie po całym kadrze rozcieńcza osobę zajmującą kilka
procent szerokokątnego widoku: na scenie 640×360 przejście postaci przez cały
kadr przesuwało średnią różnicę bezwzględną zaledwie z 0.0 na 4.0, więc każdy
sensowny próg leżał w granicach szumu. Liczone jako udział zmienionych pikseli
te same przypadki rozdzielają się na 0% / 0.9% / 5.6%.

**`min_change_fraction` to pierwszy parametr do wyregulowania na Twojej
kamerze.** Zbiera się zbyt mało klatek mimo ruchu → obniż. Zapełnia dysk
nagraniami pustego pomieszczenia → podnieś.

## Użycie

```bash
# folder ze zdjęciami (przeszukiwany rekurencyjnie)
python main.py images ./zdjecia

# plik wideo (z trackingiem i uśrednianiem embeddingów per track)
python main.py video ./nagranie.mp4 --display

# cały folder nagrań — klasteryzowane RAZEM, więc ta sama osoba
# w dwóch plikach trafia do jednej tożsamości
python main.py video ./nagrania --reset

# gęstsze próbkowanie i niższy próg dla krótkich nagrań
python main.py video ./klip.mp4 --stride 1 --min-cluster-size 2

# kamera na żywo: znane osoby podpisywane w czasie rzeczywistym,
# nieznane klasteryzowane okresowo; każda zaakceptowana twarz trafia do bazy
python main.py live --camera 0
python main.py live --camera "rtsp://user:pass@192.168.1.10/stream"
python main.py live --camera 0 --headless      # bez okna (usługa/autostart)

# ponowna klasteryzacja zapisanych embeddingów (bez ponownej detekcji)
python main.py recluster

# trening klasyfikatora online na twarzach już przypisanych do osób
python main.py train

# REST API (domyślnie 127.0.0.1:8000; dokumentacja OpenAPI pod /docs)
python main.py serve
python main.py serve --host 0.0.0.0 --port 8080
```

### Uwaga o wideo: `min_cluster_size` liczy ŚCIEŻKI, nie twarze

W trybie wideo klasteryzacja dostaje **jeden uśredniony wektor na track**, a nie
jeden na twarz. Domyślne `min_cluster_size: 4` oznacza więc, że nowa osoba musi
wystąpić w **czterech osobnych ścieżkach**, zanim powstanie dla niej tożsamość.
W krótkim nagraniu, gdzie każdy pojawia się raz, nikt nie zostanie zapisany —
przy pierwszym uruchomieniu na 60-klatkowym klipie z dwiema osobami dało to
4 ścieżki i **0 tożsamości**.

Dlatego osoby **już znane** są odzyskiwane z szumu przez kaskadę: każda próbka,
której HDBSCAN nie potrafił przypisać, jest porównywana z klasyfikatorem i
prototypami, zanim trafi do `_noise`. W tym samym teście odzyskało to wszystkie
4 ścieżki i dało 2 poprawne tożsamości. Osoba **nieznana** nadal pozostaje
szumem — kaskada nigdy nie wymyśla tożsamości.

Żeby zapisać **nową** osobę z krótkiego nagrania, obniż próg:
`--min-cluster-size 2`.

## REST API

| Metoda i ścieżka | Działanie |
|---|---|
| `POST /identify` | embedding (512D) → osoba; `commit: true` aktualizuje prototyp (EMA) |
| `POST /detect` | obraz (multipart) → wykryte twarze + identyfikacja każdej z nich |
| `GET /persons` | lista osób: etykieta, liczba twarzy, licznik embeddingów |
| `GET /persons/{id}` | pojedyncza osoba |
| `PATCH /persons/{id}` | nadanie osobie czytelnej nazwy (`{"label": "Anna"}`) |
| `DELETE /persons/{id}` | zapomnienie tożsamości (wiersze twarzy zostają — audytowalność) |
| `POST /update-prototype` | jawna aktualizacja prototypu wskazanej osoby |
| `GET /clusters` | klastry z bazy: liczność i przypisana osoba |
| `POST /retrain-classifier` | refit klasyfikatora i podmiana go w działającej kaskadzie |
| `GET /health` | stan usługi i klasyfikatora |
| `GET /metrics` | metryki Prometheusa |

`/identify` przyjmuje surowy wektor, dzięki czemu API działa **bez zainstalowanego
InsightFace** — zgodnie z podziałem na mikrousługi z `Wprowadzenie.md`, gdzie
`embedding-service` wysyła wektory do usługi tożsamości. `/detect` wymaga pełnego
stosu detekcji i zwraca `503` z czytelnym komunikatem, gdy go brakuje.

## Metryki (`/metrics`)

`fic_identifications_total{source}` (classifier/prototype/unknown),
`fic_faces_detected_total`, `fic_faces_rejected_total`, `fic_face_quality`,
`fic_identify_latency_seconds`, `fic_frame_latency_seconds` (FPS = 1/latencja),
`fic_persons_known`, `fic_clusters_total`, `fic_classifier_persons`,
`fic_classifier_trainings_total`, `fic_prototype_updates_total`.

`prometheus_client` jest opcjonalny — bez niego metryki degradują się do no-op
i reszta systemu działa bez zmian.

Wszystkie parametry w `config.yaml` (progi jakości, algorytm klasteryzacji,
progi re-identyfikacji itd.). Bez pliku konfiguracyjnego działają wbudowane
wartości domyślne.

## Wyniki (`output/`)

| Ścieżka | Zawartość |
|---|---|
| `persons/person_XXXX/` | wycinki twarzy pogrupowane per osoba |
| `persons/_noise/` | twarze nieprzypisane do żadnej osoby |
| `montages/person_XXXX.jpg` | kolaż twarzy danej osoby do szybkiej inspekcji |
| `summary.json` | statystyki: liczba osób, twarzy, jakość, źródła |
| `faces.sqlite` | wszystkie embeddingi + metadane (bbox, jakość, track, przypisanie) |
| `prototypes.json` / `.npz` | trwałe prototypy osób (re-identyfikacja między uruchomieniami) |

## Zmierzone na prawdziwych danych

Pomiary z kamery USB 1280×720, 80 zdjęć, 4 sesje (plener w pełnym słońcu,
wnętrze w słabym świetle), 2 osoby, model `buffalo_l`:

| Wielkość | Wartość |
|---|---|
| Ta sama osoba, mediana podobieństwa | 0.809 |
| Ta sama osoba, **najgorsza** para (plener vs wnętrze, profil) | **0.410** |
| Różne osoby, mediana | 0.234 |
| Różne osoby, **najlepsza** para | **0.297** |
| Margines między rozkładami | **+0.113** |
| Podobieństwo prototypów różnych osób | 0.261 |

Wnioski, które z tego płyną:

1. **Próg `match_threshold: 0.60` leży w środku przerwy** — 0.30 nad najlepszą
   parą różnych osób i 0.19 pod najgorszą parą tej samej osoby. Rozkłady się
   nie nakładają.
2. **Prototypy ratują najtrudniejsze przypadki.** Najgorsza para pojedynczych
   zdjęć tej samej osoby dała 0.410, czyli *poniżej* progu — porównanie twarz do
   twarzy uznałoby ją za dwie osoby. Uśrednienie w prototyp podniosło to samo
   porównanie do 0.82.
3. **HDBSCAN dzieli według warunków, nie tożsamości.** Cztery sesje dały 3–4
   klastry dla dwóch osób; dopiero warstwa prototypów scaliła je poprawnie.
   To nie jest usterka, tylko powód, dla którego ta warstwa istnieje.

⚠️ **Zmierzone na dwóch osobach, czyli jednej parze.** Przy 10 osobach par jest
45, przy 20 — 190, a maksimum podobieństwa między różnymi osobami rośnie z
liczbą porównań (rodzeństwo i osoby podobne potrafią osiągać 0.4–0.5).
Obserwuj tę wartość w miarę przybywania osób i obniż próg, gdy zbliży się do 0.5.

## Architektura

```
src/face_clusterer/
├── config.py       # konfiguracja (dataclasses + YAML)
├── embedder.py     # InsightFace: detekcja + alignment + embedding, L2-norm
├── quality.py      # filtr jakości: ostrość (Laplacian), rozmiar, pewność, profil
├── tracker.py      # lekki tracker IoU z fallbackiem embeddingowym (bez torch)
├── clustering.py   # HDBSCAN/DBSCAN/agglomerative + czyszczenie klastrów
├── prototypes.py   # prototypy osób: EMA, re-id, persystencja
├── classifier.py   # klasyfikator online nad embeddingami + kaskada identyfikacji
├── acquire.py      # ETAP 1: zbieranie klatek do datasetu (bez modeli)
├── scheduler.py    # nadzorca doby: zbieranie 07-15, analiza 15-07
├── api.py          # REST API (FastAPI): identyfikacja, osoby, klastry, retrain
├── metrics.py      # metryki Prometheusa (opcjonalna zależność)
├── store.py        # SQLite: embeddingi + metadane
├── report.py       # wycinki, kolaże, summary.json
└── pipeline.py     # ImagesPipeline / VideoPipeline / LivePipeline
```

### Kluczowe decyzje projektowe

1. **HDBSCAN z dwóch źródeł**: preferowany jest pakiet `hdbscan`, ale wymaga on
   kompilatora C++ i na Windowsie często nie instaluje się poprawnie — wtedy
   automatycznie używana jest implementacja z `scikit-learn >= 1.3`. Zejście od
   razu do DBSCAN po cichu odbierałoby projektowi algorytm, wokół którego jest
   zbudowany.
1. **HDBSCAN na znormalizowanych wektorach z metryką euklidesową** — biblioteka
   `hdbscan` nie wspiera metryki cosine; dla wektorów L2-znormalizowanych
   odległość euklidesowa jest monotoniczną funkcją odległości kosinusowej
   (`d_e = sqrt(2 − 2·cos)`), więc zachowanie jest równoważne.
2. **Czyszczenie klastrów po HDBSCAN** (usprawnienie względem notatek):
   HDBSCAN potrafi wciągnąć odległe punkty do klastra (gęstość jest względna) —
   członkowie o odległości kosinusowej od centroidu > `purge_threshold` są
   degradowani do szumu, a klastry mniejsze niż `min_cluster_size` usuwane.
3. **Reattach szumu**: punkty-szum dostatecznie bliskie centroidowi istniejącego
   klastra są do niego dołączane — odzyskuje poprawne twarze, wobec których
   HDBSCAN był zbyt konserwatywny.
4. **Tracking bez ciężkich zależności**: własny tracker IoU z fallbackiem po
   podobieństwie embeddingów (radzi sobie z szybkim ruchem i przeskokami przy
   dużym `frame_stride`); zamiast klasteryzować każdą klatkę, klasteryzowany
   jest jeden uśredniony (ważony jakością) embedding na track.
5. **„Dotrenowywanie" = prototypy, model zamrożony** (zgodnie z notatkami):
   model embeddingowy nie jest ruszany; uczą się wyłącznie prototypy osób
   (aktualizacja EMA) zapisywane na dysk, dzięki czemu `person_0007` z
   wczorajszego nagrania to ta sama osoba dziś.
6. **Kaskada identyfikacji: klasyfikator → prototypy → klasteryzacja**
   (warstwa 9 z `Wprowadzenie.md`). Klasyfikator jest *dotrenowywany od zera*,
   a nie przez `partial_fit` — regresja logistyczna na kilku tysiącach wektorów
   512D uczy się poniżej sekundy, a pełny refit omija główne ograniczenie
   inkrementalnych estymatorów sklearn: `partial_fit` wymaga z góry pełnej listy
   klas, czego system open-set z definicji nie ma (nowe osoby pojawiają się w
   dowolnym momencie). Klasyfikator nigdy nie tworzy osoby — poniżej progu
   pewności wstrzymuje się i decyzja spada na prototypy. Etykiety do treningu
   pochodzą z klasteryzacji i re-identyfikacji, więc trening jest
   samonadzorowany (bez adnotacji ręcznych).
8. **Stan tożsamości pod jednym zamkiem, SQLite otwierane per żądanie.**
   FastAPI uruchamia synchroniczne handlery w puli wątków, więc równoległe
   `/identify?commit=true` przeplatałyby aktualizacje EMA tego samego prototypu;
   mutacje idą przez wspólny `threading.Lock`. Połączenia SQLite nie są
   współdzielone między wątkami, więc store otwierany jest osobno w każdym
   żądaniu, a nie trzymany w stanie aplikacji.
8. **SQLite jako embedding store**: pełna historia embeddingów pozwala na
   `recluster` bez ponownej detekcji oraz budowę datasetu do ewentualnego
   offline retrainingu.

## Testy

```bash
python -m pytest tests/ -q     # 88 testów: jednostkowe, API, harmonogram, end-to-end
```

Testy nie wymagają InsightFace (embedder jest w nich zastępowany atrapą),
więc logikę można weryfikować także na maszynach bez pobranego modelu.
Testy API korzystają z `TestClient`, więc nie zajmują żadnego portu.

## Dalsza rozbudowa (zgodnie z notatkami)

- offline retraining embeddingów: triplet mining, knowledge distillation
  (warstwa 10 — wymaga GPU i zebranego datasetu)

### Errata do `Wprowadzenie.md`

Błędy techniczne znalezione w notatkach podczas audytu implementacji:

1. **Przepis na kwantyzację INT8 (sekcja Raspberry Pi) nie zadziała zgodnie z
   obietnicą.** `quantize_dynamic` z `op_types_to_quantize=[... 'Conv']` nie
   kwantyzuje warstw Conv — kwantyzacja dynamiczna w ONNX Runtime obejmuje
   operacje wagowe (MatMul/Gemm), a Conv wymaga kwantyzacji **statycznej** z
   kalibracją. ArcFace ResNet50 jest zdominowany przez Conv, więc przepis z
   notatek zostawi większość sieci w FP32 i nie da deklarowanego ~2×.
2. **Komentarz „embedding ArcFace (512D, już znormalizowany)" jest mylący.**
   `face.embedding` w InsightFace NIE jest znormalizowany (znormalizowany jest
   `face.normed_embedding`); kod w notatkach słusznie normalizuje mimo
   komentarza, a `embedder.py` robi to jawnie.
3. **Przykład pętli live w notatkach akumuluje `embeddings_buffer` bez
   ograniczenia** — na wielogodzinnym nagraniu to wyciek pamięci.
   `LivePipeline` klastruje i opróżnia bufor okresowo (`recluster_every`).
- dashboard Grafany nad metrykami z `/metrics`
- uwierzytelnianie API (obecnie brak — usługa jest przeznaczona do sieci
  wewnętrznej, stąd domyślny bind na `127.0.0.1`)
- ByteTrack/DeepSORT zamiast trackera IoU przy dużym zagęszczeniu osób
- kwantyzacja INT8 modelu ArcFace pod Raspberry Pi (przepis w `Wprowadzenie.md`)
- pgvector/Milvus zamiast SQLite przy wielu kamerach
