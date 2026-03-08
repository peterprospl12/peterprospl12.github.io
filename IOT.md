---
layout: page
title: IoT Stream Processing
permalink: /IOT/
---

## Plan: GPU-akcelerowany system przetwarzania danych strumieniowych (Real-time Stream Processing)

**TL;DR**: Budowa end-to-end systemu do analizy strumieni danych (logi sieciowe / IoT) w czasie rzeczywistym z GPU offloadingiem. Pipeline: **Kafka → Python ingestion → C++ batcher z CUDA Streams (pinned memory, async transfer) → CUDA kernele (filtrowanie, rolling stats, detekcja anomalii) → Python dashboard na żywo**. Rok czasu pozwala zbudować to solidnie — z prawdziwą detekcją anomalii, benchmarkami CPU vs GPU, i produkcyjnym monitoringiem.

---

### Kontekst: co to jest i dlaczego GPU

**Problem:** Strumienie danych z czujników IoT lub logów bezpieczeństwa generują **setki tysięcy do milionów zdarzeń na sekundę**. Każde zdarzenie wymaga: parsowania, filtrowania, obliczenia statystyk okna czasowego (rolling average, stddev), i detekcji anomalii — **w czasie rzeczywistym** (latency < 100ms).

**Dlaczego GPU:** CPU przetwarza zdarzenia sekwencyjnie lub w kilku wątkach. GPU może przetwarzać **miliony zdarzeń równolegle** — jeden wątek GPU per zdarzenie. Operacje typu "oblicz średnią i odchylenie z ostatnich 10000 punktów" to klasyczna parallel reduction. Detekcja anomalii (np. z-score, isolation) to embarrassingly parallel.

**Kluczowe wyzwanie projektowe:** Nie sam GPU kernel jest trudny — trudny jest **pipeline**: jak efektywnie przerzucać strumień danych z Kafki przez RAM do VRAM, batchować, przetwarzać i zwracać wyniki, **nie tracąc na opóźnieniach transferu** (PCIe = wąskie gardło). Tu wchodzą CUDA Streams, pinned memory, double buffering.

**Analogie produkcyjne:** NVIDIA RAPIDS cuStreamz, Apache Flink + GPU, Timeplus, ksqlDB — ale wy budujecie rdzeń od zera.

---

### Faza 0: Studia + architektura + środowisko

#### 0.1 Literatura i koncepcje do opanowania

**Stream processing fundamentals:**
- **Event time vs processing time** — zdarzenie ma timestamp kiedy powstało (event time) vs kiedy dotarło do systemu (processing time). Różnica = opóźnienie sieci/kolejki
- **Windowing** — jak grupujesz zdarzenia w okna:
  - **Tumbling window**: stałe okna bez overlapa (np. co 5 sekund)  
  - **Sliding window**: nakładające się okna (np. okno 30s co 5s)
  - **Session window**: grupowanie po aktywności (przerwa > timeout → nowa sesja)
- **Watermarking** — mechanizm: "zakładam że wszystkie zdarzenia z event_time < X już dotarły" → możesz zamknąć okno i obliczyć wynik
- **Exactly-once vs at-least-once processing** — gwarancje dostarczenia
- **Backpressure** — co jeśli GPU przetwarza wolniej niż Kafka produkuje? Mechanizm spowolnienia producenta

**GPU data transfer patterns:**
- **Pageable memory** (standardowy `malloc`) — wymaga dodatkowej kopii przez DMA staging buffer → wolny
- **Pinned memory** (`cudaMallocHost`) — DMA bezpośrednio z RAM do VRAM → **2-3× szybszy transfer**
- **CUDA Streams** — kolejki operacji GPU wykonywane asynchronicznie: memcpy i kernel mogą overlapować
- **Double/triple buffering** — podczas gdy GPU przetwarza batch N, CPU już przygotowuje batch N+1 i transferuje go → **zero idle time**
- **Zero-copy memory** (`cudaHostRegister` z flagą `cudaHostAllocMapped`) — GPU czyta bezpośrednio z RAM hosta (bez kopii) przez PCIe → dobre dla małych, rzadkich odczytów; złe dla bulk processing

**Kafka fundamentals:**
- Broker, topic, partition, consumer group, offset, commit
- Throughput: jeden partition Kafka → ~100 MB/s; wiele partitions → linear scaling
- Python client: `confluent-kafka` (C wrapper, szybki) lub `aiokafka` (async native)

**Detekcja anomalii (algorytmy do wyboru):**
- **Z-score** (najprostszy): punkt jest anomalią jeśli $|x - \mu| > k \cdot \sigma$ — wymaga rolling mean + stddev
- **Exponentially Weighted Moving Average (EWMA)**: $\mu_t = \alpha \cdot x_t + (1-\alpha) \cdot \mu_{t-1}$ — lekki, strumieniowy
- **Isolation Forest** (zaawansowany): random forest budowany na subsamplach, anomalie mają krótką ścieżkę izolacji
- **DBSCAN on GPU** (zaawansowany): density-based clustering, punkty w rzadkich regionach = anomalie
- **Autoencoder** (najbardziej zaawansowany): neural network reconstruction error jako anomaly score

#### 0.2 Architektura systemu — przegląd

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Python Layer                        │
│                                                                      │
│  ┌────────────┐    ┌───────────────┐    ┌──────────────────────────┐│
│  │ Data       │    │ Kafka         │    │ Dashboard                ││
│  │ Generator  │───→│ Producer      │    │ (Streamlit / Grafana)    ││
│  │ (fake IoT/ │    │               │    │ - live charts            ││
│  │  sec logs) │    │               │    │ - anomaly alerts         ││
│  └────────────┘    └───────┬───────┘    │ - throughput metrics     ││
│                            │            └────────────▲─────────────┘│
│                    ┌───────▼───────┐                 │              │
│                    │ Kafka         │          ┌──────┴──────┐       │
│                    │ Consumer      │          │ Results     │       │
│                    │ (python)      │          │ Consumer    │       │
│                    └───────┬───────┘          └──────▲──────┘       │
│                            │                        │              │
│                            │ raw events             │ results      │
│                            ▼                        │              │
│                    ┌───────────────────────────────────────┐       │
│                    │   Python ↔ C++ Bridge (pybind11)      │       │
│                    └───────────────────┬───────────────────┘       │
└────────────────────────────────────────┼──────────────────────────┘
                                         │
┌────────────────────────────────────────┼──────────────────────────┐
│                        C++ Integration Layer                      │
│                                                                    │
│  ┌─────────────────────────────────────▼─────────────────────────┐│
│  │                  StreamProcessor (C++)                         ││
│  │                                                               ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  ││
│  │  │ Ring Buffer   │  │ Batch        │  │ CUDA Stream        │  ││
│  │  │ (pinned mem)  │  │ Assembler    │  │ Manager            │  ││
│  │  │               │  │ (accumulate  │  │ (double buffering, │  ││
│  │  │ events land   │  │  N events    │  │  async memcpy,     │  ││
│  │  │ here first    │  │  or timeout) │  │  overlap)          │  ││
│  │  └──────┬───────┘  └──────┬───────┘  └─────────┬──────────┘  ││
│  │         └─────────────────┼─────────────────────┘             ││
│  └───────────────────────────┼───────────────────────────────────┘│
└──────────────────────────────┼────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────┐
│                       CUDA Kernels                               │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ Parse/Filter  │  │ Rolling      │  │ Anomaly Detection      │ │
│  │ Kernel        │  │ Statistics   │  │ (z-score / EWMA /      │ │
│  │ (predicate    │  │ (mean, std,  │  │  Isolation Forest)     │ │
│  │  evaluation)  │  │  min, max,   │  │                        │ │
│  │               │  │  percentile) │  │                        │ │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘ │
│  ┌──────────────┐  ┌──────────────┐                              │
│  │ Histogram /   │  │ Top-K        │                              │
│  │ Bucketing     │  │ (frequent    │                              │
│  │ Kernel        │  │  IPs, events)│                              │
│  └──────────────┘  └──────────────┘                              │
└───────────────────────────────────────────────────────────────────┘
```

#### 0.3 Wybór domeny danych

Dwie opcje (wybierzcie jedną lub obie):

**Opcja A — Logi bezpieczeństwa sieciowego (Network Security):**

Każde zdarzenie to np.:
```
{
  "timestamp": 1709901234567,        // ms od epoch
  "src_ip": "192.168.1.105",         // → uint32 (packed IP)
  "dst_ip": "10.0.0.50",             // → uint32
  "src_port": 54321,                 // uint16
  "dst_port": 443,                   // uint16
  "protocol": 6,                     // uint8 (TCP=6, UDP=17)
  "bytes_sent": 1523,                // uint32
  "bytes_recv": 48712,               // uint32
  "flags": "SYN,ACK",               // → bitmask uint8
  "event_type": "connection"         // → enum uint8
}
```

Analizy GPU:
- Filtrowanie: pokaż tylko ruch na portach 22/23/3389 (SSH/Telnet/RDP = podejrzane)
- Rolling stats: średni bytes_sent per IP w oknie 30s
- Anomalia: nagły skok bytes_sent z jednego src_ip → potencjalny data exfiltration
- Top-K: najczęściejsze src_ip→dst_ip pary (heavy hitters)
- Port scan detection: src_ip kontaktujący >N unikalnych dst_port w oknie czasowym

**Opcja B — Czujniki IoT (Industrial IoT / Smart Factory):**

```
{
  "timestamp": 1709901234567,
  "sensor_id": 42,                   // uint16
  "machine_id": 7,                   // uint16
  "temperature": 78.5,               // float32
  "vibration": 0.023,                // float32
  "pressure": 2.14,                  // float32
  "rpm": 3200,                       // uint16
  "status": "running"                // → enum uint8
}
```

Analizy GPU:
- Filtrowanie: zdarzenia gdzie temperature > próg LUB vibration > próg
- Rolling stats: średnia, stddev, min, max temperature per sensor w oknie 60s
- Anomalia: temperatura odbiega o >3σ od rolling average → alarm
- Korelacja: vibration rośnie razem z temperature? → predykcja awarii
- Histogram: rozkład RPM per maszyna w ostatniej godzinie

**Rekomendacja**: zacznij od IoT (prostsze dane, float-heavy = idealne dla GPU), potem dodaj security logs jako drugi use-case.

#### 0.4 Środowisko

- **Kafka**: Docker Compose z Kafka + ZooKeeper (lub KRaft mode) — najprostszy setup
- **CUDA Toolkit 12.x**, CMake (C++17)
- **pybind11** (Python ↔ C++ bridge)
- **Python**: confluent-kafka, asyncio, Streamlit (dashboard), matplotlib/plotly (wykresy)
- **GPU**: dowolna NVIDIA ≥ GTX 1060 (compute capability ≥ 6.0) — projekt nie jest memory-hungry
- **Nsight Systems + Nsight Compute** — profilowanie

---

### Faza 1: Data Generator + Kafka + standalone kernele 

#### 1.1 Data Generator — Python

Generator realistycznych danych z **controllable anomalies** (żebyście mogli testować detekcję):

**Normalny tryb:**
- Temperatura: random walk wokół baselinne (np. 75°C ± 2°C, Gaussian noise)
- Vibration: proporcjonalne do RPM + noise
- Logi sieciowe: Poisson arrivals, typowy mix portów (80/443 = 80%, reszta = 20%)

**Anomalie (wstrzykiwane programmatically):**
- **Spike**: temperatura skacze o +20°C na 5 sekund
- **Drift**: powolny liniowy wzrost temperatury (symuluje degradację)
- **Port scan**: jeden IP odpytuje 1000 portów w 10 sekund
- **Data exfiltration**: nagły skok bytes_sent 10× powyżej normy
- **Sensor failure**: wartość zablokowna na stałej (flatline) lub NaN

Generator pushuje do Kafki z configurowalną szybkością: 10k, 100k, 500k, 1M events/sec.

**Format danych do Kafki:**
- **Nie JSON** (za wolny do parsowania na GPU) → **binary packed struct** (Pythonowy `struct.pack`)
- Alternatywa: Apache Avro / Protocol Buffers (schema-based, kompaktowe)
- Cel: minimalizacja czasu parsowania — dane trafiają na GPU blisko raw binary

#### 1.2 Kafka Setup 

```yaml
# docker-compose.yml — Kafka + ZooKeeper
# Topics:
#   raw-events     — surowe zdarzenia od generatora
#   processed      — wyniki z GPU (statystyki, anomalie)
#   alerts         — alarmy (high priority anomalies)
```

Konfiguracja:
- Topik `raw-events`: **8–16 partitions** (parallelism), retention 1h
- Topik `processed`: 4 partitions, retention 24h
- Consumer group: `gpu-processor-group`

#### 1.3 Standalone CUDA Kernele

Każdy kernel implementowany osobno, testowany na syntetycznych danych, **zanim integracja z pipeline'em**.

**A) Filter/Predicate kernel:**

Input: tablica structów (batch of events), predykat (np. `temperature > 80.0`)
Output: indeksy zdarzeń spełniających warunek (stream compaction)

Implementacja:
- Każdy wątek GPU: ewaluuj predykat na swoim zdarzeniu
- **Stream compaction**: parallel prefix sum (scan) → scatter do output array
- Compound predicates: `temp > 80 AND vibration > 0.05` — jeden kernel, wiele warunków
- Użyj `cub::DeviceSelect::If` jako reference/porównanie

**B) Rolling Statistics kernel (sliding window):**

To najtrudniejszy kernel — wymaga utrzymywania stanu między batchami.

Podejście 1 — **Full window in memory:**
- Utrzymuj ring buffer na GPU z ostatnimi W zdarzeniami per sensor
- Kernel: per sensor, oblicz mean/stddev/min/max z ring buffera
- Problem: O(W) per sensor per batch — drogi dla dużych okien

Podejście 2 — **Incremental (Welford's algorithm):**
- Utrzymuj running state: `{count, mean, M2}` per sensor na GPU
- Nowe zdarzenie: update online: $\text{mean}_n = \text{mean}_{n-1} + \frac{x_n - \text{mean}_{n-1}}{n}$, $M2_n = M2_{n-1} + (x_n - \text{mean}_{n-1})(x_n - \text{mean}_n)$
- Stddev = $\sqrt{M2 / n}$
- Problem: nie obsługuje "wygaśnięcia" starych danych (infinite window)
- Rozwiązanie: **Exponentially Weighted** — automatycznie degraduje stare dane

Podejście 3 — **Segmented reduction z timestampami:**
- Dane posortowane per sensor wg timestamp
- Kernel: per sensor, binary search na okno [now - W, now], parallel reduction na tym segmencie
- Najczystszy koncept, dobry dla tumbling windows

**Rekomendacja**: zacznij od EWMA (podejście 2 — najprostsze, strumieniowe), potem dodaj full sliding window (podejście 1/3) jako upgrade.

**C) Anomaly Detection kernel — Z-score:**
- Input: nowe zdarzenie + running stats (mean, stddev per sensor)
- Oblicz: $z = \frac{|x - \mu|}{\sigma}$
- Output: `anomaly_score` per zdarzenie + boolean `is_anomaly` (z > threshold, np. 3.0)
- Kernel: embarrassingly parallel — jeden wątek per zdarzenie

**D) Anomaly Detection — EWMA (Exponentially Weighted Moving Average):**
- State per sensor: `ewma_mean`, `ewma_var`
- Update: $\mu_t = \alpha x_t + (1-\alpha)\mu_{t-1}$, analogicznie wariancja
- Anomaly: $|x_t - \mu_{t-1}| > k \cdot \sqrt{\text{var}_{t-1}}$
- Szybki, strumieniowy, nie wymaga okna — idealny dla IoT

**E) Histogram / Bucketing kernel:**
- Input: tablica wartości float (np. temperatures)
- Output: histogram z `num_bins` kubełków
- Implementacja: `atomicAdd` na globalnym histogramie (ok dla umiarkowanej ilości binów)
- Opcja: per-block partial histogram w shared memory → global merge (szybsza dla dużej ilości binów)

**F) Top-K / Heavy Hitters kernel:**
- Input: tablica kluczy (np. IP adresów jako uint32)
- Output: K najczęściej występujących kluczy
- Implementacja: GPU hash table (count per key) → parallel top-K extraction
- Alternatywa: Count-Min Sketch na GPU (probabilistyczny, O(1) space per key)

**G) Reduction kernele (global aggregation):**
- SUM, COUNT, MIN, MAX, MEAN per kolumna per batch
- Standard parallel reduction (CUB library jako reference)

**Testowanie (krytyczne!):**
- Każdy kernel: generuj dane w Python, oblicz reference wynik (NumPy/Pandas), porównaj z GPU
- Edge cases: puste batche, NaN/Inf values, single-element batch, very large batch (10M+)
- Floating point: porównuj z tolerancją (redukcja zmienia kolejność → rounding differences)

---

### Faza 2: Integration Layer — CUDA Streams, Pinned Memory, Batching 

**To jest serce projektu z punktu widzenia systemowego** 

#### 2.1 Pinned Memory Ring Buffer

**Problem:** standardowy transfer CPU→GPU przez pageable memory:
```
CPU pageable mem → CPU pinned staging → DMA → GPU VRAM
```
To dwa kroki. Pinned memory eliminuje pośredni krok → **2-3× szybszy transfer.**

**Implementacja:**
- Alokuj ring buffer w pinned memory: `cudaMallocHost(&buffer, BUFFER_SIZE)`
- Kafka consumer wpisuje surowe eventy do ring buffera (zero-copy z perspektywy CPU)
- Batch assembler czyta z ring buffera i grupuje po `BATCH_SIZE` lub `TIMEOUT_MS`
- Ring buffer musi być thread-safe: producent (Kafka thread) i konsument (batch assembler) → lock-free SPSC (Single Producer Single Consumer) queue

**Parametry do tuningu:**
- `BATCH_SIZE`: ile eventów w jednym batchu (np. 4096, 16384, 65536)
- `TIMEOUT_MS`: max czas czekania na zebranie pełnego batcha (np. 10ms-100ms)
- Trade-off: duży batch = lepszy GPU throughput, ale wyższy latency; mały batch = niski latency, ale GPU underutilized

#### 2.2 CUDA Streams + Double Buffering

**Kluczowa optymalizacja** — overlap transfer z compute:

```
Timeline bez overlap (naive):
  [Transfer batch 1] [Compute batch 1] [Transfer batch 2] [Compute batch 2] ...
  
Timeline z double buffering + 2 CUDA streams:
  Stream A: [Transfer 1] [         Compute 1          ] [Transfer 3] ...
  Stream B:              [Transfer 2] [   Compute 2    ]             ...
  
  → Transfer batch N+1 dzieje się RÓWNOCZEŚNIE z compute batch N!
```

**Implementacja:**
- Alokuj 2 (lub 3) bufory w GPU memory — "ping-pong buffers"
- 2 CUDA Streams: `stream_a`, `stream_b`
- Pipeline per iteration:
  1. `cudaMemcpyAsync(gpu_buf_a, cpu_batch, size, cudaMemcpyHostToDevice, stream_a)`
  2. `launch_kernel<<<grid, block, 0, stream_a>>>(gpu_buf_a, ...)`
  3. `cudaMemcpyAsync(cpu_results, gpu_results_a, size, cudaMemcpyDeviceToHost, stream_a)`
  4. **Równocześnie** na stream_b z następnym batchem
- `cudaStreamSynchronize()` przed odczytem wyników z CPU-side

**Triple buffering (opcjonalnie):**
- 3 bufory, 3 streamy → jeszcze lepsze overlapowanie, szczególnie gdy H2D, compute, i D2H mają różne czasy

#### 2.3 Batch Assembler + Event Struct

**Definicja structu danych (C++ side):**
```
Koncepcja (Structure of Arrays — SoA, lepsze dla GPU):
  
Batch (SoA layout):
  float*    temperatures    [BATCH_SIZE]
  float*    vibrations      [BATCH_SIZE]      
  float*    pressures       [BATCH_SIZE]
  uint16_t* sensor_ids      [BATCH_SIZE]
  int64_t*  timestamps      [BATCH_SIZE]
  int       count           // ile eventów w batchu (≤ BATCH_SIZE)
```

**Dlaczego SoA zamiast AoS (Array of Structs):**
- AoS: `Event[N]` — wątki GPU czytają `events[tid].temperature` → non-coalesced (przeskakuje cały struct)
- SoA: `temperatures[N]` — wątki czytają `temps[tid]` → **coalesced** (sąsiednie wątki czytają sąsiednie adresy) → **5-10× szybszy memory access na GPU**

**Batch assembler logic:**
1. Czytaj eventy z ring buffera
2. Transpozycja AoS → SoA (na CPU, w pinned memory)
3. Kiedy batch gotowy (count == BATCH_SIZE lub timeout) → trigger async transfer

#### 2.4 pybind11 Bridge — Python ↔ C++

Eksponuj do Pythona:
- `class StreamProcessor`:
  - `__init__(config: dict)` — inicjalizacja GPU, alokacja buforów, tworzenie streamów
  - `submit_batch(data: np.ndarray)` → wrzuć batch do pipeline'u (async)
  - `get_results() → dict` → odbierz przetworzone wyniki (statystyki, anomalie)
  - `get_stats() → dict` → metryki: throughput, latency, GPU utilization
  - `shutdown()` → cleanup
- Lifecycle: Python consumer → dekoduje Kafkowe bajty → `submit_batch()` → C++ przejmuje

**Alternatywne podejście (zero-copy NumPy):**
- Python consumer buduje NumPy array z eventów
- pybind11 bierze NumPy buffer pointer → `cudaMemcpyAsync` direct z NumPy data buffer (jeśli pinned via `cudaMallocHost`)
- Unika dodatkowej kopii

#### 2.5 State Management — rolling statistics across batches

**Problem**: kernele rolling stats potrzebują stanu (mean, stddev per sensor) który **przeżywa między batchami**.

**Rozwiązanie**: persistent state buffers na GPU:
- Alokuj `float* state_mean[NUM_SENSORS]`, `float* state_var[NUM_SENSORS]` na GPU raz na start
- Każdy batch: kernel czyta stary state → update z nowymi danymi → zapisuje nowy state
- Stan nigdy nie wraca na CPU (chyba że checkpoint/snapshot)
- **Checkpoint**: co N batchów, `cudaMemcpy` stanu na CPU → zapis na dysk → recovery po crashu

---

### Faza 3: Zaawansowana detekcja anomalii + streaming features 

Rok czasu pozwala pójść dalej niż prosty z-score.

#### 3.1 Wielowymiarowa detekcja anomalii na GPU

**A) Mahalanobis distance (multi-variate z-score):**
- Zamiast z-score per feature osobno → uwzględnij korelacje między features
- $D^2 = (x - \mu)^T \Sigma^{-1} (x - \mu)$ gdzie $\Sigma$ = macierz kowariancji
- Wymaga: running covariance matrix per sensor group (6×6 matrix for 6 features) → update online
- GPU kernel: batch matrix multiply per event (małe macierze → warp-level)

**B) GPU Isolation Forest (zaawansowane, opcjonalne):**
- Build: random subsampling → random feature, random split → drzewo decyzyjne
- Score: ścieżka od korzenia do liścia → krótka ścieżka = anomalia
- **GPU**: batch traversal wielu drzew równolegle — każdy wątek przechodzi jedno drzewo dla jednego punktu
- Literature: "GPU-accelerated Isolation Forest" — istnieją implementacje reference

**C) Autoencoder (micro neural network) — opcjonalnie
- Mały FC autoencoder (input → 32 → 8 → 32 → input)
- Train na "normalnych" danych → anomalia = wysoki reconstruction error
- Inference na GPU: zwykły forward pass (GEMV × 4 warstwy) → ultra szybki
- Train: offline na historical data lub online z exponential forgetting

#### 3.2 Sliding Window na GPU (full implementation)

Poprawa vs EWMA — prawdziwe sliding window z dokładnymi statystykami:

**Segmented Circular Buffer approach:**
- Per sensor: circular buffer na GPU `float[WINDOW_SIZE]`
- Nowy event: nadpisz najstarszy slot → oblicz stats z pełnego bufora
- Kernel: per sensor, parallel reduction na circular buffer → mean, stddev, min, max, percentile

**Efficient percentile (p50, p95, p99):**
- Sortowanie per-sensor window → `cub::BlockRadixSort`
- Alternatywa: approximate with T-Digest / DDSketch (streaming quantile sketches)

#### 3.3 Complex Event Processing (CEP)

Wykrywanie **wzorców w sekwencjach zdarzeń** — nie single-event anomaly, ale patterns:

- **Port scan**: src_ip kontaktuje >N unikalnych dst_port w oknie T → GPU: per-IP hash set, count threshold
- **Brute force attack**: >N failed logins z tego samego IP → GPU: counting per IP + time window
- **Correlated anomaly**: temperature spike NA sensor A FOLLOWED BY vibration spike NA sensor B w ciągu T sekund → requires cross-sensor temporal logic

Implementacja: state machine per entity (IP / sensor) na GPU, aktualizacja per batch.

#### 3.4 Multi-GPU / Multi-stream scaling

Jeśli jeden GPU to za mało:
- **Partition by key**: sensor_id % num_gpus → każdy GPU przetwarza podzbiór sensorów
- **Replicated pipeline**: multiple CUDA streams na jednym GPU z niezależnymi batchami
- Kafka consumer group: wiele procesów, każdy z innym GPU

---

### Faza 4: Dashboard + Python frontend 

#### 4.1 Real-time Dashboard 

**Streamlit dashboard (najprostsze, wystarczające):**

Panele:
1. **Live Throughput**: events/sec processed (linia czasowa, ostatnie 5 min)
2. **Latency**: p50/p95/p99 end-to-end latency (Kafka ingestion → GPU result)
3. **Rolling Stats per Sensor**: heatmapa temperature per sensor over time
4. **Anomaly Feed**: tabela: timestamp, sensor_id, value, anomaly_score, type
5. **Anomaly Rate**: % zdarzeń flagowanych jako anomalia (tumbling window 10s)
6. **GPU Utilization**: memory used, kernel occupancy, transfer throughput

**Alternatywa — Grafana + InfluxDB/Prometheus:**
- GPU processor exportuje metryki do Prometheus (via `prometheus_client`)
- Grafana dashboard z auto-refresh
- Bardziej "produkcyjne", ale wyższy setup cost

#### 4.2 Alert System

- Anomalia score > HIGH_THRESHOLD → push do Kafka topic `alerts`
- Python alert consumer → webhook / email / Slack notification
- Rate limiting: max 1 alert per sensor per minute (deduplikacja)

#### 4.3 Results Format

GPU processor zwraca per batch:
```python
{
    "batch_id": 12345,
    "timestamp_range": [start_ts, end_ts],
    "stats": {
        "sensor_42": {"mean": 78.2, "std": 1.3, "min": 75.1, "max": 82.4},
        ...
    },
    "anomalies": [
        {"timestamp": ..., "sensor_id": 42, "value": 112.3, "score": 4.7, "type": "spike"},
        ...
    ],
    "global": {"events_processed": 65536, "anomaly_count": 3, "processing_time_ms": 2.1}
}
```

---

### Faza 5: Profilowanie, benchmarki, optymalizacja 

#### 5.1 Metryki wydajności

| Metryka | Co mierzy | Target |
|---|---|---|
| **Throughput** (events/sec) | Ile zdarzeń GPU przetwarza per sekundę | >1M events/sec na RTX 3090 |
| **E2E latency** (ms) | Kafka ingestion → wynik dostępny | <50ms (p95) |
| **GPU compute time** (ms) | Sam czas kerneli per batch | <5ms per batch 64K |
| **Transfer time** (ms) | CPU→GPU + GPU→CPU per batch | <2ms per batch 64K |
| **Transfer overlap** (%) | Ile transferu overlapuje z compute | >80% |
| **GPU memory** | Zużycie VRAM | Baseline + per-sensor state + bufory |
| **CPU overhead** | Kafka consumer + Python overhead | <20% of E2E |

#### 5.2 CPU vs GPU Comparison (kluczowe!)

Zaimplementuj **ten sam pipeline** w czystym CPU (Python + NumPy/Pandas):
- Te same algorytmy (z-score, EWMA, rolling stats)
- Ta sama logika batchowania
- Mierz: throughput, latency

Porównanie: GPU speedup = CPU_time / GPU_time per batch (oczekiwany: **10-100× dla dużych batchów**)

Dodatkowe porównanie:
- **CPU C++** (bez GPU) — fair comparison, ten sam algorytm w C++ na CPU
- **CPU multithreaded (OpenMP)** — ile daje wielowątkowość CPU vs GPU
- **cuStreamz (RAPIDS)** — jeśli dostępny, porównaj z GPU-native streaming framework

#### 5.3 Scaling experiments

| Eksperyment | Zmienna | Co mierzysz |
|---|---|---|
| Batch size sweep | batch: 1K, 4K, 16K, 64K, 256K, 1M | Throughput i latency vs batch size |
| Num sensors sweep | sensors: 100, 1K, 10K, 100K | Throughput vs state complexity |
| Window size sweep | window: 100, 1K, 10K, 100K | Impact on rolling stats kernel |
| Event rate sweep | rate: 10K, 100K, 500K, 1M, 5M/sec | Max sustainable throughput |
| Feature count sweep | dims: 2, 4, 8, 16 | Impact multivariate anomaly |

#### 5.4 Narzędzia profilowania

- **Nsight Systems**: timeline — widać overlap streams, kerneli i memcpy
- **Nsight Compute**: per-kernel analiza — occupancy, bandwidth utilization, warp stalls
- **Kafka monitoring**: lag per partition (czy GPU nadąża za producentem?)
- Custom timing: `cudaEvent_t` start/stop wokół każdego etapu

#### 5.5 Optymalizacja na podstawie profili

Typowe bottlenecks i rozwiązania:
| Bottleneck | Rozwiązanie |
|---|---|
| Transfer dominuje | Zwiększ batch size; verify pinned memory; triple buffering |
| Kernel niski occupancy | Zmniejsz register usage, adjust block size |
| Kernel memory-bound | SoA layout; coalesced access; shared memory cache |
| Kafka consumer lag | Więcej partitions; C++ consumer (librdkafka) zamiast Python |
| Python overhead | Minimize Python↔C++ round trips; batch results |

---

### Faza 6: Praca pisemna + finalizacja 

#### 6.1 Struktura pracy

| Rozdział | Zawartość |
|---|---|
| 1. Wstęp | Motywacja: real-time analytics, GPU vs CPU, stream processing |
| 2. Przegląd literatury | Stream processing (Flink, Kafka Streams), GPU computing, anomaly detection |
| 3. Architektura systemu | Diagram pipeline, komponenty, flow danych |
| 4. Implementacja kerneli CUDA | Filter, rolling stats, anomaly detection — algorytmy + optymalizacje GPU |
| 5. Warstwa integracji CPU-GPU | Pinned memory, CUDA Streams, double buffering, SoA layout, pybind11 |
| 6. Pipeline przetwarzania | Kafka integration, batching strategy, state management, dashboard |
| 7. Ewaluacja eksperymentalna | Throughput/latency, CPU vs GPU, scaling, profiling results |
| 8. Podsumowanie | Wnioski, ograniczenia, future work |

---

### Podział odpowiedzialności — szczegółowy

#### Inżynieria Danych / Architektura (Python)

**Zakres:**
- Data generator z controllable anomalies (IoT + opcjonalnie security logs)
- Kafka setup (Docker Compose), topics, configuration
- Kafka producer (binary packed events → topic `raw-events`)
- Kafka consumer (topic `raw-events` → decode → `StreamProcessor.submit_batch()`)
- Results consumer (topic `processed` → dashboard)
- Real-time dashboard (Streamlit): live charts, anomaly feed, throughput metrics
- Alert system (anomaly → topic `alerts` → notification)
- CPU baseline implementation (Python + NumPy) — do porównania z GPU

**Kluczowe deliverables:**
- Generator produkujący 1M events/sec do Kafki
- Dashboard pokazujący live wyniki
- CPU baseline z identyczną logiką

**Czego się uczy:**
- Kafka ecosystem (producer/consumer, partitions, offsets, consumer groups)
- Binary serialization (`struct.pack`, Protocol Buffers/Avro)
- Async Python (`asyncio`, `aiokafka`)
- Real-time visualization (Streamlit auto-refresh / WebSocket)

---

#### GPU / Kernel (CUDA C/C++)

**Zakres:**
- Filter/predicate kernel (stream compaction)
- Rolling statistics kernel (EWMA + opcjonalnie full sliding window)
- Anomaly detection kernele: z-score, EWMA-based, opcjonalnie Mahalanobis / Isolation Forest
- Histogram / bucketing kernel
- Top-K / heavy hitters kernel
- Global reduction (SUM, COUNT, MIN, MAX)
- FP32 precision, NULL/NaN handling
- Unit testy kerneli (vs NumPy reference)
- Profilowanie kerneli z Nsight Compute

**Kluczowe deliverables:**
- Zestaw poprawnych, wydajnych kerneli z testami
- Raport profilowania: occupancy, memory bandwidth, roofline
- Dokumentacja parametrów każdego kernela

**Czego się uczy:**
- CUDA programming model: grid, block, warp, thread
- Parallel reduction, scan (prefix sum), stream compaction
- Atomic operations, shared memory, coalesced access
- Online algorithms na GPU (Welford, EWMA)
- Nsight Compute profilowanie

---

#### Inżynieria Pamięci i Integracja (C++ / pybind11)

**Zakres — najtrudniejsza i najciekawsza rola:**
- Ring buffer w pinned memory (`cudaMallocHost`)
- Batch assembler (AoS → SoA transpozycja, batch by count or timeout)
- CUDA Stream Manager (double/triple buffering, async memcpy, stream sync)
- Persistent state management (rolling stats state na GPU, checkpointing)
- pybind11 bridge: `StreamProcessor` klasa Python → C++
- Memory pool (unikanie powtarzanych `cudaMalloc/Free`)
- Zero-copy experiments (benchmarking `cudaHostAllocMapped` vs pinned + copy)
- Error handling i recovery (GPU error → restart stream, Kafka re-seek)

**Kluczowe deliverables:**
- `StreamProcessor` C++ class z pełnym pipeline'em
- pybind11 wrapper działający z Python
- Demonstracja: overlap transfer + compute na timeline (Nsight Systems screenshot)
- Benchmark: pinned vs pageable, double vs single buffer, batch size tuning

**Czego się uczy:**
- Low-level memory management (pinned, mapped, managed memory)
- CUDA Streams i async operations
- CPU-GPU communication patterns
- Lock-free data structures (ring buffer)
- C++ → Python binding (pybind11, lifetime management, GIL)
- Double/triple buffering pattern

---

#### Analiza wydajności   

**Zakres:**
- CPU baseline w C++ (OpenMP) — fair comparison
- Benchmark suite (automatyczne testy: batch size sweep, rate sweep, sensor count sweep)
- Nsight Systems profiling: timeline analysis całego pipeline
- Nsight Compute profiling: per-kernel deep dive
- CPU vs GPU comparison (tabele, wykresy)
- Scaling analysis (throughput vs batch size, vs sensor count, vs feature count)
- Bottleneck identification i rekomendacje optymalizacji
- Kafka lag monitoring (czy GPU nadąża?)
- Redakcja pracy magisterskiej
- Przegląd literatury (Related Work chapter)

**Kluczowe deliverables:**
- Komplet benchmarków: tabele throughput/latency, wykresy scaling
- CPU (Python) vs CPU (C++) vs GPU porównanie
- Nsight timeline screenshots z analizą overlap
- Draft pracy magisterskiej

**Czego się uczy:**
- Performance engineering methodology
- GPU profiling tools (Nsight Systems/Compute)
- Roofline model analysis
- Scientific benchmarking (warm-up, repetitions, confidence intervals)

---
### Ryzyka i mitygacja

| Ryzyko | Prawdop. | Mitygacja |
|---|---|---|
| Kafka setup problematyczny na Windows | Średnie | Docker; alternatywa: RabbitMQ lub Redis Streams (prostsze) |
| Batch size za mały → GPU underutilized | Średnie | Tuning sweep; minimum 4K events; adaptive batching |
| CUDA Streams nie overlapują (dependency) | Niskie | Verify z Nsight Systems timeline; ensure independent streams |
| Rolling stats state corruption (race conditions) | Średnie | Per-sensor isolation (jeden wątek per sensor); atomics |
| Python GIL blokuje przy pybind11 | Niskie | Release GIL w C++ (`py::gil_scoped_release`); async submit |
| Kafka consumer lag (GPU za wolny) | Niskie | Backpressure mechanism; increase batch size |

---

### Kluczowe zasoby

1. **"CUDA C++ Programming Guide"** (NVIDIA) — rozdziały: Streams, Pinned Memory, Async Operations
2. **"Kafka: The Definitive Guide"** (O'Reilly) — fundamenty
3. **CUB Library docs** — DeviceReduce, DeviceSelect, DeviceRadixSort
4. **Nsight Systems User Guide** — timeline profiling
5. **RAPIDS cuStreamz** — reference GPU streaming: github.com/rapidsai/custreamz
6. **pybind11 docs** — szczególnie: numpy integration, GIL management, shared_ptr
7. **Chandola et al., "Anomaly Detection: A Survey"** (ACM Computing Surveys 2009) — przegląd algorytmów

---

### Dalsze decyzje

1. **Kafka vs prostszy broker**: Kafka to standard, ale RabbitMQ lub Redis Streams łatwiejsze w setup. Kafka lepiej na CV.
2. **Domena**: IoT (prostsze) vs Security Logs (ciekawsze, ale stringi = trudniejsze na GPU). Można obie.
3. **Zakres anomaly detection**: czy z-score + EWMA wystarczy, czy iść w Isolation Forest / Autoencoder?


Kafka **nie musi być**. I szczerze — lepiej architektonicznie jest zrobić to jako **abstrakcyjny interfejs**, gdzie Kafka to tylko jeden z możliwych adapterów.

### Dlaczego to lepsze podejście

1. **Sedno projektu to GPU processing, nie Kafka admin** — nie chcecie spędzić 3 tygodni na debugowaniu Docker + ZooKeeper zamiast pisać kernele
2. **Testowalność** — standalone testy kerneli i pipeline'u bez odpalonego brokera
3. **Elastyczność** — ten sam silnik GPU działa z czymkolwiek: Kafka, RabbitMQ, Redis Streams, plik CSV, socket TCP, albo generator in-memory
4. **Lepiej wygląda w pracy** — "architektura z pluggable data source" to solidniejszy design niż "hardcoded Kafka consumer"

### Jak to zaprojektować

Interfejs w C++ (lub Python — zależy od warstwy):

```
Koncepcja:

class DataSource (abstrakcja):
    fetch_batch(max_events, timeout_ms) → Batch of events

Implementacje:
    InMemoryGenerator    — do testów i benchmarków (najważniejszy!)
    KafkaSource          — jeśli chcecie Kafkę
    SocketSource         — TCP/UDP socket (prosty network stream)
    FileReplaySource     — odtwarzanie z zapisanego pliku (reproducible benchmarks)
    ZMQSource            — ZeroMQ (lekki broker, zero setup)
```

### Co rekomenduję zamiast Kafki jako default

| Opcja | Setup | Kiedy użyć |
|---|---|---|
| **InMemoryGenerator** (Python/C++) | Zero setup | Development, benchmarki, testy kerneli — **80% czasu to będziecie używać** |
| **ZeroMQ (ZMQ)** | `pip install pyzmq`, zero infra | Jeśli chcecie network source bez brokera — producer/consumer przez socket |
| **Redis Streams** | Jeden `docker run redis` | Jeśli chcecie broker, ale prostszy niż Kafka |
| **Kafka** | Docker Compose, ZooKeeper/KRaft, topics, partitions... | Jeśli chcecie "produkcyjny" look na CV |
| **Plain TCP socket** | Zero dependencies | Najprostszy network transport |

### Jak to zmienia plan

- `InMemoryGenerator` — z controllable anomalies (to samo co wcześniej, ale bez Kafki)
- `DataSource` interface w Pythonie
- Opcjonalnie: jeden adapter (ZMQ lub Redis) jako "network" demo
- Dashboard (Streamlit) — czyta wyniki bezpośrednio z `StreamProcessor.get_results()`
- CPU baseline

Pipeline upraszcza się:

```
PRZED (z Kafką):
  Generator → Kafka → Consumer → pybind11 → C++ batcher → GPU → Kafka → Dashboard

PO (interfejs):
  DataSource.fetch_batch() → pybind11 → C++ batcher → GPU → callback/queue → Dashboard
```

Mniej ruchomych części = mniej bugów = więcej czasu na GPU. 

Kafka możecie dodać **pod koniec** (mies. 9-10) jako jeden z adapterów — "our system also supports Kafka ingestion" — jeden rozdział w pracy, ale nie fundamentalna zależność.

Chcesz żebym zaktualizował pełny plan z tym podejściem interface-based?