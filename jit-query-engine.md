---
layout: page
title: JIT Query Engine
permalink: /jit-query-engine/
---

## Plan: GPU-akcelerowany JIT Query Engine na dane tabelaryczne

**TL;DR**: Budowa od zera silnika analitycznego, który przyjmuje zapytania DataFrame/SQL-like, kompiluje je w runtime (NVRTC) do zoptymalizowanych **fused kerneli CUDA** (eliminując materializację pośrednią), i wykonuje na GPU. Trzy domeny w jednym: **GPU programming + bazy danych + kompilatory**. Rok czasu pozwala zrobić to solidnie — z operator fusion, cost-based plannerem, i realnym porównaniem z cuDF/DuckDB/Polars.

---

### Kontekst: co to jest i dlaczego jest trudne

**Problem:** Silniki analityczne (DuckDB, Polars) przetwarzają dane tabelaryczne operator-po-operatorze: filter → materializuj wynik → group_by → materializuj → aggregate. Każda "materializacja" to zapis/odczyt z pamięci — **wąskie gardło na GPU**, gdzie bandwidth jest drogą walutą.

**Rozwiązanie — operator fusion:** Zamiast osobnych kerneli na filter, group_by, agg — **generujesz jeden fused kernel** CUDA, który robi wszystko w jednym przejściu przez dane. Nie ma pośredniej materializacji. Tak działa NVIDIA RAPIDS (cuDF) pod spodem, Hyper (Tableau), i częściowo Photon (Databricks).

**Co dokładnie budujesz:**
1. **Standalone kernele** operatorów relacyjnych na GPU (filter, hash join, group-by, sort)
2. **JIT code generator** — z drzewa zapytania generuje kod CUDA, kompiluje via NVRTC w runtime
3. **Operator fusion** — łączy pipeline operatorów w jeden kernel
4. **Query planner** — decyduje jaki plan fizyczny jest optymalny (kolejność joinów, kiedy fuzjować, kiedy materializować)
5. **Python DataFrame API** — lazy evaluation → automatyczna kompilacja → wykonanie na GPU
6. **Benchmarki na TPC-H** — standard branżowy

---

### Faza 1: Fundamenty — studia + columnar storage + standalone kernele

#### 1.1 Studia teoretyczne 

**Architektura silników analitycznych — co trzeba zrozumieć:**
- **Volcano/Iterator model** vs **Vectorized execution** vs **Compiled (data-centric)** model
  - Volcano: `next()` na operatorze zwraca tuple — powolny (virtual calls, brak locality)
  - Vectorized (DuckDB, Velox): `next()` zwraca wektor/batch ~1024 wierszy — amortyzuje overhead
  - **Compiled/data-centric** (HyPer, twój projekt): kompiluj zapytanie do tight loop przetwarzający dane — **najszybszy**, to wasz cel
- **Paper "Efficiently Compiling Efficient Query Plans for Modern Hardware"** (Thomas Neumann, VLDB 2011) — FUNDAMENT. Opisuje push-based compilation, pipeline breakers, produce/consume model
- **Paper "Everything You Always Wanted to Know About Compiled and Vectorized Queries But Were Afraid to Ask"** (Kersten et al., VLDB 2018) — porównanie podejść
- **NVIDIA RAPIDS cuDF** — source code, szczególnie JIT compilation path
- **Paper "GPU-Accelerated Database Systems: Survey and Open Challenges"** — przegląd

**Koncepcje GPU do opanowania:**
- **NVRTC API** — runtime compilation CUDA source → PTX → cubin. Kluczowe: `nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetPTX`, potem `cuModuleLoadData` + `cuLaunchKernel` z Driver API
- **Columnar vs row storage na GPU** — columnar = coalesced access = szybki na GPU
- **GPU hash tables** — open addressing, cuckoo hashing, Robin Hood — fundamentalne dla join i group-by

**Kluczowe source code do przestudiowania:**
- DuckDB (C++) — `src/execution/`, `src/optimizer/`, `src/planner/`
- Polars (Rust) — lazy evaluation, query plan optimization
- cuDF — `cpp/src/join/`, `cpp/src/groupby/`, `python/cudf/cudf/core/`
- HeavyDB (dawniej OmniSci/MapD) — GPU-native database, open source

#### 1.2 Columnar storage na GPU 

**Kolumnowy format danych — absolutna podstawa:**

Tabela w pamięci GPU to zestaw kolumn, gdzie każda kolumna to ciągły bufor:
```
Table "orders":
  Column "price":    float32[] na GPU  (1 cudaMalloc na wektor)
  Column "quantity":  int32[] na GPU
  Column "category": int32[] na GPU   (dictionary-encoded strings)
  Column "date":     int64[] na GPU   (timestamp)
  Validity bitmask:  uint8[] per column (NULL handling)
```

**Co zaimplementować:**
- `class GpuColumn` — type-erased kolumna: pointer do GPU buffer, dtype (INT32/INT64/FLOAT32/FLOAT64/STRING_DICT), length, null bitmask
- `class GpuTable` — kolekcja kolumn + schema (nazwy, typy)
- Transfer CPU ↔ GPU: efektywne `cudaMemcpy` z/do NumPy arrays, Apache Arrow compatibility (opcjonalnie)
- **Dictionary encoding** dla stringów: stringi → tablica unikalna + int32 indices → GPU operuje na int32 
- **NULL handling**: bitmask (1 bit per row), każdy kernel musi respektować nulle

#### 1.3 Standalone kernele operatorów

Najpierw implementujesz każdy operator jako **osobny, ręcznie napisany kernel CUDA** (bez JIT). To daje wam baseline poprawności i wydajności.

**A) Filter (selection) kernel:**
- Input: kolumna danych + predykat (np. `price > 100.0`)
- Output: indeksy spełniające warunek LUB skompaktowana kolumna
- Implementacja: **stream compaction** — scan (parallel prefix sum) + scatter
- GPU pattern: każdy wątek ewaluuje predykat na swoim wierszu, potem parallel prefix sum daje output pozycję
- **Compound predicates**: `price > 100 AND category == 3` — ewaluacja wszystkich warunków w jednym kernelu
- Benchmark: porównaj z `thrust::copy_if`

**B) Hash Join kernel:**
- Jedno z najtrudniejszych — hash join na GPU to aktywny obszar badawczy
- **Build phase**: zbuduj hash table z mniejszej tabeli (build side)
  - GPU hash table: open addressing z linear/quadratic probing
  - Alokacja: prealokowany bufor 2× rozmiar build side (load factor ~0.5)
  - Hash function: MurmurHash3 lub xxHash (dobre rozkłady na GPU)
- **Probe phase**: dla każdego wiersza z większej tabeli (probe side), lookup w hash table
  - Każdy wątek GPU: hash(key) → probe slot → porównaj → output match
- **Output materialization**: problem — nie znamy rozmiaru output z góry!
  - Rozwiązanie 1: two-pass — pass 1 liczy ile matchów (atomic counter per wą­tek), pass 2 zapisuje
  - Rozwiązanie 2: pre-allocated output z upper bound
- Obsługa: inner join, left join (z fallback na NULL)

**C) Group-by Aggregation kernel:**
- **Hash-based aggregation**: hash table gdzie key = group-by columns, value = running aggregate
- Aggregaty: SUM, COUNT, AVG (= SUM+COUNT), MIN, MAX
- Implementacja: `atomicAdd` dla SUM/COUNT, `atomicMin/Max` dla MIN/MAX
- Problem z `atomicAdd` na float: ograniczona precyzja — rozwiązanie: **per-warp partial aggregation** w shared memory, potem jeden atomic per warp
- Multi-column group-by: hash composite key (concat + hash)

**D) Sort kernel:**
- **Radix sort** — najszybszy na GPU (O(n) dla fixed-width keys)
- Użyj `cub::DeviceRadixSort` (CUB library) jako baseline
- Opcjonalnie: własna implementacja per-block radix sort + merge
- Potrzebny do ORDER BY i sort-merge join

**E) Aggregation bez group-by (global agg):**
- Parallel reduction: SUM, COUNT, MIN, MAX, AVG na całej kolumnie
- GPU pattern: block-level reduction w shared memory → atomic na global result
- Użyj `cub::DeviceReduce` jako reference

**Testy poprawności (krytyczne!):**
- Dla KAŻDEGO kernela: generuj losowe dane, wykonaj na GPU, porównaj wynik z Pandas/Polars na CPU
- Edge cases: puste tabele, NULL values, duplicate keys, skewed data distribution
- Numeryczna dokładność: FP32 reduction ordering → porównaj z double-precision reference

---

### Faza 2: JIT Compilation Pipeline + Query Planner 

To jest **intelektualne jądro projektu** — tu zmieniasz się z "napisałem kernele" w "zbudowałem kompilator zapytań".

#### 2.1 Intermediate Representation — drzewo zapytania 

**Logical Plan (drzewo operatorów):**
```
Zapytanie: df.filter(col("price") > 100).group_by("category").agg(sum("price"), count())

Logical Plan:
  Aggregate(group_by=["category"], aggs=[Sum("price"), Count()])
    └── Filter(predicate: col("price") > 100.0)
          └── Scan("orders")
```

**Co zaimplementować (C++):**
- `class LogicalNode` — bazowa klasa z wariantami:
  - `ScanNode` — źródło danych (tabela GPU)
  - `FilterNode` — predykat (expression tree)
  - `ProjectNode` — wybór/transformacja kolumn
  - `JoinNode` — typ joina + klucze
  - `AggregateNode` — group-by keys + aggregate expressions
  - `SortNode` — klucze sortowania + ASC/DESC
  - `LimitNode` — LIMIT/OFFSET
- `class Expression` — drzewo wyrażeń:
  - `ColumnRef("price")`, `Literal(100.0)`, `BinaryOp(GT, left, right)`, `AggFunc(SUM, col)`
  - Operatory: `+`, `-`, `*`, `/`, `>`, `<`, `==`, `!=`, `AND`, `OR`, `NOT`

**Physical Plan:**
- `PhysicalNode` — konkretna implementacja:
  - `HashJoinPhysical` vs `SortMergeJoinPhysical` (różne strategie)
  - `HashAggregatePhysical` vs `SortAggregatePhysical`
  - `GpuFilterPhysical` — compilable to kernel

#### 2.2 Code Generator — NVRTC

**Serce systemu.** Generujesz string z kodem CUDA na podstawie drzewa zapytania, kompilujesz w runtime.

**Pipeline:**
```
Logical Plan → Physical Plan → CUDA source string → NVRTC compile → PTX → cubin → launch kernel
```

**Kluczowy koncept — produce/consume model (z paperu Neumanna):**

Każdy operator "produkuje" wiersze i "konsumuje" je od dziecka. Przy code generation:
- **Scan** produkuje: generuje pętlę `for (int tid = ...; tid < n; tid += stride)`
- **Filter** konsumuje od Scan, produkuje dalej: generuje `if (predykat)` wewnątrz pętli
- **Aggregate** konsumuje: generuje `atomicAdd` / hash table insert

**Przykład wygenerowanego kodu:**
```
Zapytanie: filter(price > 100) → group_by(category) → sum(price)

Wygenerowany kernel (schemat):
__global__ void fused_query_kernel(
    float* col_price, int* col_category, int n,
    HashTable* ht, float* agg_sum) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    // --- Filter (produce) ---
    float price = col_price[tid];
    if (!(price > 100.0f)) return;  // predykat!
    
    // --- Aggregate (consume) ---
    int category = col_category[tid];
    int slot = hash_probe(ht, category);
    atomicAdd(&agg_sum[slot], price);
}
```

**Zauważ**: filter + aggregate w **jednym kernelu**, bez materializacji pośredniej! To jest cały punkt fusion.

**Co zaimplementować:**
- `class CodeGenerator` (C++):
  - `visit(ScanNode)` → generuj pętlę i ładowanie kolumn
  - `visit(FilterNode)` → generuj `if (predykat)` 
  - `visit(AggregateNode)` → generuj hash table operations
  - `visit(ProjectNode)` → generuj obliczenia wyrażeń
  - `generateKernelSource() → string` — pełny kernel CUDA jako tekst
- `class ExpressionCodeGen` — tłumaczenie drzewa wyrażeń na kod C:
  - `col("price") > 100` → `col_price[tid] > 100.0f`
  - `col("a") + col("b") * 2` → `col_a[tid] + col_b[tid] * 2`
- `class NVRTCCompiler`:
  - `compile(source_string) → CUmodule`
  - Cache skompilowanych modułów (hash source → cubin) — żeby nie rekompilować tego samego zapytania
  - Error handling: NVRTC compilation errors → czytelny komunikat

#### 2.3 Pipeline Breakers i strategia fuzji 

**Nie wszystko da się sfuzjować!** "Pipeline breakers" to operatory, które MUSZĄ zmaterializować wyniki:

| Operator | Pipeline breaker? | Dlaczego |
|---|---|---|
| Filter | Nie | Generuje `if` w pętli |
| Project (wyrażenia) | Nie | Generuje obliczenia inline |
| Hash Join (probe side) | Nie* | Probe to lookup — fuzjuje się |
| Hash Join (build side) | **TAK** | Musi zbudować hash table PRZED probe |
| Group-by Aggregate | **TAK** | Musi zbudować hash table z agregatami |
| Sort | **TAK** | Inherently materialization-heavy |
| Limit | Nie (jeśli po sort) | Obcięcie | 

**Strategia:**
- Podziel plan na **pipeline segments** oddzielone breakers
- Każdy segment → jeden fused kernel
- Między segmentami → materializacja (zapis do GPU memory)
- Przykład: `Scan → Filter → HashJoin(probe) → Aggregate`
  - Segment 1 (build side): osobny kernel budujący hash table
  - Segment 2 (fused): Scan → Filter → Probe → Aggregate w jednym kernelu

#### 2.4 Query Planner / Optimizer 

**Logical → Physical plan transformation z optymalizacjami:**

**Rule-based optimizations (na początek):**
- **Predicate pushdown**: przenieś filter jak najniżej (bliżej scan) — filtruj wcześnie, mniej danych w pipeline
- **Projection pushdown**: ładuj tylko potrzebne kolumny (mniej bandwidth)
- **Join ordering** (dla wielu joinów): mniejsza tabela jako build side
- **Constant folding**: `col("x") > 3 + 2` → `col("x") > 5` (w compile time, nie runtime)

**Cost-based optimization (zaawansowane):**
- Statystyki tabel: row count, cardinality per column, min/max, histogram
- Cost model: szacuj koszt operacji GPU (ilość danych × bandwidth, hash table build cost)
- Wybór: HashJoin vs SortMergeJoin na podstawie rozmiaru danych
- Wybór: fuzjować czy materializować (jeśli intermediate result mały — materializacja tańsza niż complex fused kernel z dużym register pressure)

#### 2.5 Hash Table na GPU — komponent współdzielony 

Potrzebny dla Join i Group-by. To **osobny, kluczowy komponent**.

**Implementacja:**
- Open addressing z linear probing (najprostsza, dobra na GPU — coalesced probing)
- Prealokowany bufor: `Entry[capacity]` gdzie `Entry = {key, value, occupied}`
- **Capacity**: 2× expected elements (load factor 0.5)
- **Hash function**: MurmurHash3 finalizer (szybki, dobry rozkład)
- **Insert**: `atomicCAS` na slot → jeśli zajęty, linear probe next
- **Probe**: sekwencyjne sprawdzanie slotów od hash(key)
- **Multi-column keys**: hash composite (pack do int64 lub hash-combine)
- **Warp-cooperative probing** (zaawansowane): cały warp szuka razem → lepszy throughput

---

### Faza 3: Python API + zaawansowane optymalizacje 

#### 3.1 Python DataFrame API z lazy evaluation

**Wzorzec: Polars-style lazy API**

Użytkownik pisze:
```python
import gpu_query as gq

df = gq.read_parquet("orders.parquet")  # transfer do GPU

result = (
    df
    .filter(gq.col("price") > 100)
    .filter(gq.col("region") == "EU")
    .join(categories_df, on="category_id", how="inner")
    .group_by("category_name")
    .agg(
        gq.col("price").sum().alias("total_revenue"),
        gq.col("price").mean().alias("avg_price"),
        gq.col("order_id").count().alias("num_orders")
    )
    .sort("total_revenue", descending=True)
    .limit(10)
    .collect()  # ← TU dopiero odpala kompilację + wykonanie!
)
```

**Kluczowe**: do momentu `.collect()` nic się nie wykonuje — buduje się tylko drzewo logicznego planu. `collect()` triggeruje: optimize → plan → codegen → compile → execute → zwróć wynik.

**Implementacja (pybind11):**
- `class LazyFrame` (Python):
  - Każda metoda (`filter`, `join`, `group_by`, `sort`) zwraca **nowy LazyFrame** z rozszerzonym drzewem planu
  - `collect()` → wywołuje C++ engine: `optimize(plan) → codegen(plan) → compile() → execute() → GpuTable → konwersja do Pandas/PyArrow`
- `class Expr` (Python): `gq.col("x")`, `gq.lit(42)`, operatory `+`, `-`, `>`, `==` — budują drzewo wyrażeń
- `class AggExpr`: `.sum()`, `.mean()`, `.count()`, `.min()`, `.max()`

**Zwracanie wyników:**
- `collect() → pandas.DataFrame` (domyślnie — kopiuj z GPU do CPU)
- `collect(to="arrow") → pyarrow.Table` (zero-copy jeśli ArrowCUDA)
- `collect(to="gpu") → GpuDataFrame` (zostaw na GPU — do chainowania z innymi GPU operacjami)

#### 3.2 I/O — ładowanie danych

- **CSV reader**: Python-side (pandas) → NumPy → `cudaMemcpy` → GpuTable
- **Parquet reader**: `pyarrow.parquet.read_table()` → Arrow → GPU (szybsze, kolumnowy format)
- **Opcjonalnie**: GPU-native Parquet reader (jak cuDF) — ale to osobny duży projekt, skip
- **Random data generator**: do benchmarków — generuj TPC-H tabele programowo

#### 3.3 Zaawansowane optymalizacje 

**A) Multi-column fusion i wyrażenia złożone:**
- `filter(col("a") > 10 AND col("b") + col("c") < col("d") * 2)` → jeden predykat w kernelu
- Arytmetyka w projection: `select(col("price") * col("qty") - col("discount"))` → fused z resztą pipeline

**B) Cost-based optimizer z statystykami:**
- Zbieraj statystyki przy load: row_count, cardinality (count distinct), min/max per column
- Histogramy (equi-width) dla lepszego szacowania selektywności filtrów
- Decyzje: join ordering, hash join vs sort-merge join, kiedy materializować

**C) Kernel cache:**
- Hash source code → przechowuj skompilowany cubin
- Parametryzacja: ten sam schemat zapytania z różnymi literałami → ten sam kernel, różne argumenty launch
- LRU eviction policy na cache

**D) Memory management:**
- **Memory pool** (arena allocator) na GPU — unikanie kosztownego `cudaMalloc` per operację
- Inspiracja: RMM (RAPIDS Memory Manager) — suballocator z `cudaMalloc` na dużych chunkach
- Tracking GPU memory usage, OOM handling (spill to CPU)

**E) String operations (opcjonalne):**
- Dictionary encoding z GPU-side string comparison
- LIKE / contains / starts_with — kernel operujący na char arrays
- To jest trudne i opcjonalne — ale dodaje dużo wartości

**F) Window functions (opcjonalne):**
- `ROW_NUMBER()`, `RANK()`, `SUM() OVER (PARTITION BY ... ORDER BY ...)`
- Wymaga sorted partitions → sort-based approach
- Ambitne, ale publikowalne

---

### Faza 4: TPC-H Benchmarki + praca pisemna 

#### 4.1 TPC-H Benchmark

**TPC-H** to standard branżowy do testowania silników analitycznych. 8 tabel (lineitem, orders, customer, supplier, nation, region, part, partsupp), 22 zapytania.

**Skala danych:**
- SF=1 (1 GB) — development, debugging
- SF=10 (10 GB) — główne benchmarki
- SF=100 (100 GB) — test skalowania (jeśli GPU ma dość pamięci, lub out-of-GPU-memory handling)

**Które zapytania zaimplementować (gradacja trudności):**

| TPC-H Query | Operatory | Trudność |
|---|---|---|
| **Q1** (pricing summary) | Scan → Filter → Group-by → Agg(SUM,AVG,COUNT) | Łatwe — idealny pierwszy test |
| **Q6** (forecasting revenue) | Scan → Filter → Agg(SUM) | Najłatwiejsze — pure filter+agg |
| **Q3** (shipping priority) | 3-way Join → Filter → Group-by → Agg → Sort | Średnie — test join + agg |
| **Q5** (local supplier volume) | Multi-join (6 tabel!) → Group-by → Agg → Sort | Trudne — test join ordering |
| **Q9** (product type profit) | Multi-join → Group-by (expression) → Agg → Sort | Trudne — expressions w group-by |
| **Q12** (shipping modes) | Join → Filter → Group-by → Agg (CASE WHEN) | Średnie — conditional aggregation |
| **Q14** (promotion effect) | Join → Agg (CASE WHEN expression) | Średnie |
| **Q19** (discounted revenue) | Join → complex OR predicate → Agg | Średnie — compound predicates |

**Rekomendacja**: zaimplementuj Q1, Q6, Q3, Q5, Q12, Q14 — pokrywa filter, join, agg, sort, expressions.

#### 4.2 Metryki benchmarkowe

| Metryka | Co mierzy |
|---|---|
| **Query execution time** (ms) | End-to-end per query (excl. data load) |
| **Compilation time** (ms) | NVRTC compile — ile trwa JIT? |
| **Throughput** (GB/s) | Ile danych przetwarza per sekundę (bandwidth utilization) |
| **Memory usage** (MB) | Peak GPU memory per query |
| **Fusion benefit** | Czas fused kernel vs materialized pipeline (A/B test!) |
| **Speedup vs CPU** | Porównanie z DuckDB/Polars na tym samym zapytaniu |

#### 4.3 Porównanie z istniejącymi systemami

| System | Opis | Porównanie |
|---|---|---|
| **cuDF (RAPIDS)** | GPU DataFrame (NVIDIA) | Bezpośredni competitor — GPU native |
| **DuckDB** | CPU analityczny, vectorized | Baseline CPU — tu mierzysz GPU speedup |
| **Polars** | CPU, Rust, lazy eval | Kolejny baseline CPU |
| **HeavyDB** | GPU-native database | Jeśli dostępny — najbliższy odpowiednik |
| **Twój silnik (interpreted)** | Standalone kernele BEZ fusion | Mierzy wartość dodaną JIT fusion |

**Najważniejsze porównanie**: **twój silnik z fuzją** vs **twój silnik BEZ fuzji** (materialized intermediate) — to izoluje dokładnie ile daje fusion.

#### 4.4 Narzędzia profilowania
- **Nsight Systems** — timeline: compile time, kernel launch, memcpy, overlap
- **Nsight Compute** — per-kernel: occupancy, memory throughput, compute utilization, roofline
- **nvidia-smi** — memory tracking
- **Custom instrumentation** — timer wokół każdego etapu (parse, optimize, codegen, compile, execute)

#### 4.5 Praca pisemna — struktura

| Rozdział | Zawartość |
|---|---|
| 1. Wstęp | Motywacja: GPU dla analityki, problem materializacji |
| 2. Przegląd literatury | Neumann (compiled queries), RAPIDS, GPU hash joins, vectorized vs compiled |
| 3. Architektura systemu | Diagram pipeline, columnar storage, JIT pipeline |
| 4. Implementacja kerneli | Filter, join, agg, sort na GPU — optymalizacje |
| 5. JIT compilation i fusion | Code generator, NVRTC, pipeline breakers, przykłady generowanego kodu |
| 6. Query planner | Rule-based + cost-based optimization |
| 7. Python API | Lazy evaluation, DataFrame interface |
| 8. Ewaluacja eksperymentalna | TPC-H results, porównania, analiza fusion benefit |
| 9. Podsumowanie | Wnioski, ograniczenia, future work |

---

### Podział odpowiedzialności (4 osoby)

| Osoba | Zakres | Kluczowe deliverables |
|---|---|---|
| **A — Kernele CUDA (operatory relacyjne)** | Filter (scan+predicate+compaction), Hash Join (build+probe), Group-by Agg (hash-based, atomics), Sort (radix), Global Agg. GPU hash table. NULL handling. FP32/FP64/INT32/INT64. Profilowanie Nsight Compute. | Poprawne kernele z testami vs Pandas, raport profilowania roofline |
| **B — JIT Code Generator + Planner (C++)** | Logical/Physical plan nodes, Expression trees, CodeGenerator (produce/consume), NVRTC compilation, kernel cache, pipeline breaker detection, operator fusion logic. Rule-based optimizer (predicate pushdown, projection pruning). Cost-based optimizer (statystyki, join ordering). | Działający pipeline: plan → source → compile → execute. Demo: zapytanie Q1 TPC-H fused vs unfused. |
| **C — Python DataFrame API + I/O** | pybind11 LazyFrame/Expr wrappery, lazy evaluation chain, `.collect()` trigger, Parquet/CSV reader, Apache Arrow interop, TPC-H data generator, NumPy/Pandas kompatybilność, memory pool (arena allocator), dokumentacja API. | Działający Python API: `df.filter().group_by().agg().collect()` zwracający Pandas DataFrame. |
| **D — Benchmarki + dokumentacja + praca** | TPC-H benchmark suite (Q1,Q3,Q5,Q6,Q12,Q14), porównanie z cuDF/DuckDB/Polars, fusion A/B test, Nsight profiling, wizualizacja query plans, generowanie wykresów, redakcja pracy magisterskiej, przegląd literatury. | Komplet wykresów, tabela porównawcza, draft pracy. |

---

### Harmonogram roczny (szczegółowy)

| Miesiąc | Osoba A (Kernele) | Osoba B (JIT/Planner) | Osoba C (Python/IO) | Osoba D (Benchmarki) |
|---|---|---|---|---|
| **1** | Studia: GPU patterns, hash tables | Studia: Neumann paper, compilation | Studia: pybind11, Arrow, Polars API | Studia: TPC-H, DuckDB arch, narzędzia |
| **2** | GpuColumn/GpuTable, Filter kernel | Logical plan nodes, Expression tree | Szkielet Python: LazyFrame, Expr | TPC-H data generator (SF=1) |
| **3** | Hash Join kernel, Sort kernel | Physical plan nodes, basic visitor | CSV/Parquet loader → GPU | Testy poprawności kerneli vs Pandas |
| **4** | Group-by Agg kernel, GPU hash table | CodeGenerator v1 (filter+agg fusion) | pybind11 bindings: GpuTable ↔ Python | Benchmark standalone kerneli |
| **5** | Optymalizacja kerneli, NULL handling | NVRTC compiler, kernel cache | LazyFrame: filter/project/collect chain | Fusion A/B test (filter+agg) |
| **6** | Hash Join + probe fusion | Pipeline breaker detection, full fusion | Join/group_by/sort w LazyFrame API | TPC-H Q1, Q6 benchmarki |
| **7** | Multi-type support (INT/FLOAT/INT64) | Rule-based optimizer (pushdown) | Arrow interop, memory pool | TPC-H Q3 + porównanie z DuckDB |
| **8** | **String ops** (dict encoding, opcja) | **Cost-based optimizer** (statystyki) | **Window functions** (opcja) | TPC-H Q5, Q12, Q14 |
| **9** | Profilowanie i tuning wszystkich kerneli | Tuning planner, edge cases | Polish API, error handling, docs | **Pełne porównanie** z cuDF/DuckDB/Polars |
| **10** | Bugfix, edge cases, final optymalizacje | Bugfix, edge cases | Demo notebook (Jupyter) | **Redakcja pracy** — rozdziały 1-4 |
| **11** | Support dla benchmarków D | Support dla benchmarków D | Support dla benchmarków D | **Redakcja pracy** — rozdziały 5-8 |
| **12** | Code review, cleanup | Code review, cleanup | Code review, README, packaging | **Finalizacja pracy** — rozdział 9, korekty |

---

### Architektura systemu — przegląd

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Layer                           │
│  ┌───────────┐  ┌────────────┐  ┌─────────────────────────┐│
│  │ LazyFrame  │  │ Expr /     │  │ I/O: CSV, Parquet,     ││
│  │ API        │  │ AggExpr    │  │ Arrow, NumPy            ││
│  │ .filter()  │  │ col("x")   │  │ read_parquet()          ││
│  │ .join()    │  │ .sum()     │  │                         ││
│  │ .group_by()│  │ > < == +   │  │                         ││
│  │ .collect() │  │            │  │                         ││
│  └─────┬─────┘  └─────┬──────┘  └──────────┬──────────────┘│
│        └───────────────┼────────────────────┘               │
│                        │  pybind11                          │
└────────────────────────┼────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      C++ Engine                             │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐ │
│  │ Logical Plan │───→│ Optimizer    │───→│ Physical Plan │ │
│  │ (AST)        │    │ (rules +    │    │               │ │
│  │              │    │  cost-based) │    │               │ │
│  └──────────────┘    └──────────────┘    └───────┬───────┘ │
│                                                   │         │
│  ┌────────────────────────────────────────────────▼───────┐ │
│  │              Code Generator                            │ │
│  │  Physical Plan → Pipeline Segments → CUDA source code  │ │
│  └────────────────────────────┬───────────────────────────┘ │
│                               │                             │
│  ┌────────────────────────────▼───────────────────────────┐ │
│  │              NVRTC Compiler + Cache                    │ │
│  │  CUDA source → PTX → cubin → CUmodule                 │ │
│  └────────────────────────────┬───────────────────────────┘ │
│                               │                             │
│  ┌────────────────────────────▼───────────────────────────┐ │
│  │              Execution Engine                          │ │
│  │  Launch fused kernels, manage GPU memory, collect      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Memory Manager (Arena/Pool Allocator)                 │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    CUDA Kernels                             │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Filter   │ │ Hash Join │ │ Group-by │ │ Sort (Radix) │ │
│  │ (stream  │ │ (build +  │ │ Agg      │ │              │ │
│  │ compact) │ │  probe)   │ │ (hash +  │ │              │ │
│  │          │ │           │ │  atomics)│ │              │ │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────┘ │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GPU Hash Table (open addressing, linear probing)    │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  JIT-generated Fused Kernels (NVRTC runtime)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

### Proponowana struktura repozytorium

```
gpu-query-engine/
├── CMakeLists.txt
├── csrc/
│   ├── kernels/
│   │   ├── filter.cu                ← stream compaction kernel
│   │   ├── hash_join.cu             ← build + probe kernels
│   │   ├── hash_table.cuh           ← GPU open-addressing hash table
│   │   ├── aggregate.cu             ← group-by + global aggregation
│   │   ├── sort.cu                  ← radix sort wrapper (cub)
│   │   └── reduction.cu             ← parallel reduction (SUM, MIN, MAX)
│   ├── storage/
│   │   ├── gpu_column.cpp/.hpp      ← type-erased columnar storage
│   │   ├── gpu_table.cpp/.hpp       ← collection of columns + schema
│   │   └── memory_pool.cpp/.hpp     ← arena allocator for GPU memory
│   ├── plan/
│   │   ├── logical_plan.cpp/.hpp    ← AST nodes (Scan, Filter, Join, Agg, Sort)
│   │   ├── physical_plan.cpp/.hpp   ← physical operators
│   │   ├── expression.cpp/.hpp      ← expression tree (col, lit, binop, agg)
│   │   └── optimizer.cpp/.hpp       ← rule-based + cost-based optimizer
│   ├── codegen/
│   │   ├── code_generator.cpp/.hpp  ← plan → CUDA source (produce/consume)
│   │   ├── nvrtc_compiler.cpp/.hpp  ← NVRTC compilation + cache
│   │   └── kernel_cache.cpp/.hpp    ← source hash → compiled cubin LRU cache
│   ├── execution/
│   │   ├── executor.cpp/.hpp        ← launch kernels, manage pipeline
│   │   └── query_engine.cpp/.hpp    ← top-level: plan → optimize → compile → run
│   └── bindings/
│       └── python_bindings.cpp      ← pybind11: LazyFrame, Expr, Engine
├── python/
│   ├── gpu_query/
│   │   ├── __init__.py
│   │   ├── lazy_frame.py            ← LazyFrame Python wrapper
│   │   ├── expr.py                  ← col(), lit(), operators
│   │   ├── io.py                    ← read_csv, read_parquet
│   │   └── config.py                ← GPU device selection, memory limits
│   └── benchmarks/
│       ├── tpch_datagen.py          ← TPC-H data generator
│       ├── tpch_queries.py          ← Q1, Q3, Q5, Q6, Q12, Q14
│       ├── benchmark_runner.py      ← automated benchmark suite
│       └── compare_systems.py       ← run same queries on cuDF/DuckDB/Polars
├── tests/
│   ├── test_kernels.py              ← kernel correctness vs Pandas
│   ├── test_codegen.py              ← generated code correctness
│   ├── test_optimizer.py            ← plan transformations
│   ├── test_lazy_frame.py           ← Python API E2E
│   └── test_tpch.py                 ← TPC-H query results vs DuckDB reference
└── docs/
    ├── thesis/
    └── examples/
        └── demo_notebook.ipynb      ← Jupyter demo
```

---

### Ryzyka i mitygacja

| Ryzyko | Prawdop. | Mitygacja |
|---|---|---|
| GPU hash join daje wrong results (race conditions) | Wysokie | Extensive testing z małymi danymi, porównanie z Pandas merge na 100% danych |
| NVRTC compile time za wolny (>100ms per query) | Średnie | Kernel cache (skip compile jeśli ten sam query), parametryzacja literałów |
| GPU memory nie mieści TPC-H SF=10 | Średnie | Chunked processing (przetwarzaj dane w kawałkach), lub testuj na SF=1 |
| Code generator produkuje niepoprawny CUDA | Wysokie (na start) | Unit testy na wygenerowany kod, porównanie wyników z unfused baseline |
| Fusion nie daje speedupu (kernel za złożony → niski occupancy) | Niskie | Profilowanie Nsight → tuning, fallback do unfused jeśli fusion wolniejszy |
| Python API za wolne (overhead pybind11) | Niskie | pybind11 overhead znikomy; bottleneck to GPU execution |

---

### Kluczowe papery i zasoby

1. **Neumann, "Efficiently Compiling Efficient Query Plans for Modern Hardware"** (VLDB 2011) — fundamentalny paper o kompilacji zapytań, produce/consume model
2. **Kersten et al., "Everything You Always Wanted to Know About Compiled and Vectorized Queries"** (VLDB 2018) — porównanie podejść
3. **Shanbhag et al., "Efficient Top-K Query Processing on Massively Parallel Hardware"** — GPU query processing
4. **Paul et al., "GPU-Accelerated Database Systems: Survey and Open Challenges"** — przegląd
5. **Sioulas et al., "Hardware-Conscious Hash-Joins on GPUs"** (ICDE 2019) — hash join na GPU
6. **NVIDIA RAPIDS cuDF** — source code: github.com/rapidsai/cudf
7. **DuckDB** — source code: github.com/duckdb/duckdb (reference CPU analityczny)
8. **TPC-H specification** — tpc.org (schema tabel, zapytania)
9. **NVRTC documentation** — NVIDIA Runtime Compilation
10. **CUB library** — GPU primitives (scan, sort, reduce) — cub.github.io

---

### Dalsze pytania / decyzje

1. **Ile osób w zespole?** Plan zakłada 4. Da się ściąć do 3 (łącząc C+D) lub 2 (A robi kernele+codegen, B robi Python+benchmarki).
2. **Scope joinów**: czy multi-way join (3+ tabel) jest wymagany, czy wystarczą binary joins?
3. **String operations**: dictionary encoding wystarczy, czy potrzebne LIKE / regex?
4. **Docelowa skala TPC-H**: SF=1 (szybkie testy) czy SF=10+ (poważne benchmarki)?