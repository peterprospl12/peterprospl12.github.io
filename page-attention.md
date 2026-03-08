## Plan: GPU-akcelerowany silnik inferencji LLM z PagedAttention i ciągłym batchowaniem

**TL;DR**: Budowa od zera silnika serwowania LLM w CUDA C++ i Pythonie, implementującego dwie kluczowe innowacje: **PagedAttention** (wirtualizacja pamięci KV-cache) i **continuous batching** (dynamiczne zarządzanie requestami). Działający prototyp obsługujący GPT-2 → Llama-2-7B, z OpenAI-compatible API.

---

### Faza 0: Studia teoretyczne i przygotowanie

**Literatura obowiązkowa:**
1. **"Efficient Memory Management for LLM Serving with PagedAttention"** (Kwon et al., SOSP 2023) — fundament projektu
2. **"Orca"** (Yu et al., OSDI 2022) — iteration-level scheduling = continuous batching  
3. **FlashAttention** (Dao et al.) — tiling attention (kontekst, nie implementujecie FlashAttention, ale rozumiecie trade-offy)
4. Kod źródłowy **vLLM** — szczególnie `csrc/attention/`, `vllm/core/scheduler.py`, `vllm/core/block_manager.py`

**Co musicie zrozumieć przed kodowaniem:**

**A) KV-cache w LLM inference:**
- **Prefill phase**: cały prompt naraz → compute-bound (GEMM)
- **Decode phase**: token po tokenie → memory-bound (GEMV), tu KV-cache rośnie
- KV-cache przechowuje wektory Key/Value z poprzednich tokenów per warstwa × per głowica
- Problem: Llama-2-7B, seq_len=2048, batch=32 → ~32 GB KV-cache samo na cache!

**B) Dlaczego PagedAttention:**
- Tradycyjnie: ciągły blok pamięci na `max_seq_len` per request → **60-80% pamięci marnowane** (fragmentacja wewnętrzna)
- PagedAttention: KV-cache dzielony na "strony" (bloki ~16 tokenów), alokowane dynamicznie → ~zero fragmentacji, memory sharing (np. wspólny system prompt)

**C) Continuous batching vs static:**
- Static: czekasz na najdłuższy request → krótkie marnują czas
- Continuous: po każdym decode step — kto skończył wychodzi, nowy wchodzi z kolejki
- **Preemption**: brak pamięci → swap KV-cache na CPU, wznów później

**Środowisko:** CUDA 12.x, CMake (C++17), pybind11, PyTorch (do walidacji), HF transformers+safetensors, GPU: RTX 3090/4090 lub A100

---

### Faza 1: Kernele CUDA — PagedAttention ← NAJWAŻNIEJSZA

**1.1 Struktury danych KV-cache na GPU:**
- `KVBlock`: tablica `K[block_size][num_heads][head_dim]` + `V[...]` — block_size typowo 16 tokenów
- `PageTable` (per request): `int32[]` mapujący logical_block → physical_block
- `BlockAllocator`: prealokowana pula bloków (jeden wielki `cudaMalloc`), thread-safe free list

**1.2 Kernel PagedAttention (decode phase) — SERCE PROJEKTU:**

Algorytm per request, per head:
1. Załaduj query vector `q ∈ R^{head_dim}` (nowy token)
2. Iteruj po blokach KV via page table:
   - Załaduj `K[block_size × head_dim]`, oblicz `scores = K @ q / sqrt(d)`
   - **Online softmax** (running max + sum — stabilność numeryczna bez dwóch passów)
3. Oblicz weighted sum V z normalizowanymi wagami
4. Zapisz output ∈ R^{head_dim}

**Kluczowe optymalizacje:**
- Warp-level reduction (dot product q·k_i — jeden warp per blok KV)
- Vectorized loads (`float4` / `half2` dla coalesced access)
- Shared memory — cache query vector (czytany wielokrotnie)
- Grid: `(num_seqs, num_heads, ceil(max_blocks))` lub pętla po blokach

**Walidacja:** output vs `torch.nn.functional.scaled_dot_product_attention`, tolerancja <1e-3 FP16

**1.3 Prefill attention** — pragmatycznie: użyj cuBLAS GEMM do Q@K^T + softmax + attn@V, lub FlashAttention jako dependency

**1.4 Fused GEMV** — decode linear layers: albo custom kernel (tiled po out_dim), albo **cuBLAS** `cublasHgemv` (wystarczy na tezę)

**1.5 Helper kernele:** RoPE (rotary embeddings), RMSNorm, SiLU+gate multiply, top-p/top-k sampling

---

### Faza 2: Scheduler + Memory Manager w C++

**2.1 Request State Machine:**
```
WAITING → RUNNING → (SWAPPED) → FINISHED
```
Każdy `SequenceGroup`: prompt_token_ids, output_token_ids, page_table, status, sampling_params

**2.2 Block Manager:**
- `allocate(seq, num_blocks)` — alokuj fizyczne bloki, aktualizuj page table
- `free(seq)` — po zakończeniu generacji
- `swap_out(seq)` → `cudaMemcpyAsync` GPU→CPU (kopiuj KV-cache na host)
- `swap_in(seq)` → CPU→GPU (wznów request)
- Opcjonalnie: **Copy-on-Write** dla wspólnych prefixów

**2.3 Scheduler — wywoływany co iteration:**
1. Kto skończył (EOS / max_tokens)? → FINISHED, zwolnij bloki
2. Running requests potrzebują nowego bloku? → jeśli brak wolnych: **preemption** (swapuj najstarszy)
3. Czy swapowane requesty mogą wrócić? → swap_in
4. Kolejka WAITING → dodaj nowe do batcha (ile się zmieści)
5. Zwróć `SchedulerOutput` (kto jedzie, co swap'ować)

**2.4 Jeden krok inference (orchestracja):**
1. `scheduler.schedule()` → output
2. Swap in/out (async memcpy)
3. Zbierz input batch: tokeny, page tables, pozycje
4. Forward pass: embedding → (RMSNorm → Q/K/V proj → RoPE → **PagedAttention** → output proj → RMSNorm → FFN → residual) × num_layers → LM head → logits
5. Sampling → nowe tokeny, alokuj nowe bloki
6. Sprawdź stop conditions

---

### Faza 3: Model Loader + Forward Pass

- Parsowanie `config.json` z HF → num_layers, hidden_size, num_heads, head_dim, GQA num_kv_heads
- Ładowanie wag z `.safetensors` → konwersja FP16 → upload na GPU
- Forward pass: cuBLAS dla linear layers + custom kernele dla attention/norm/activation
- **GQA support**: Llama-2/Mistral mają mniej KV heads niż Q heads → broadcast mapping

**Model docelowy:** zacznij od **GPT-2 (124M)** — mały, szybki iteracje, MHA. Potem **Llama-2-7B** — GQA, RoPE, SiLU.

---

### Faza 4: Python API + integracja 

- **pybind11 wrapper**: klasa `LLMEngine` z metodami `load_model()`, `add_request()`, `step()`
- **FastAPI server** z endpointami OpenAI-compatible:
  - `POST /v1/completions`, `POST /v1/chat/completions`
  - **SSE streaming** (token-by-token)
- Async engine loop (background `engine.step()` w pętli)
- Integracja: HF tokenizer, model download z HF Hub, chat templates

---

### Faza 5: Benchmarki i praca pisemna

**Metryki:**
| Metryka | Co mierzy |
|---|---|
| **TTFT** (Time To First Token) | Latency prefill |
| **TPOT** (Time Per Output Token) | Decode speed |
| Tokens/sec | Throughput |
| GPU memory utilization | Efektywność PagedAttention |
| p50/p95/p99 latency | Tail latency under load |

**Scenariusze testowe:** single request (baseline), fixed batch (throughput scaling), online serving (Poisson arrivals), memory pressure (test swapping/preemption), **paged vs contiguous KV A/B test**

**Porównanie z:** vLLM, HF TGI, TensorRT-LLM — ten sam model, hardware, prompty

**Narzędzia:** Nsight Systems (timeline), Nsight Compute (kernel analysis), custom async load generator

---
---

### Proponowana struktura repozytorium

```
llm-engine/
├── csrc/
│   ├── kernels/          ← paged_attention.cu, rope.cu, rmsnorm.cu, activation.cu, sampling.cu
│   ├── model/            ← llama.cpp/hpp, model_loader.cpp/hpp
│   ├── core/             ← scheduler.cpp/hpp, block_manager.cpp/hpp, sequence.cpp/hpp
│   └── bindings/         ← python_bindings.cpp (pybind11)
├── python/
│   ├── llm_engine/       ← engine.py, server.py, sampling_params.py, tokenizer.py
│   └── benchmarks/       ← benchmark_throughput.py, load_generator.py
├── tests/                ← test_paged_attention.py, test_scheduler.py, test_e2e.py
├── CMakeLists.txt
└── docs/thesis/
```

---

### Ryzyka i pragmatyczne skróty

| Ryzyko | Mitygacja |
|---|---|
| Kernel daje wrong results | Testy vs PyTorch reference na małych inputach, bit-level comparison |
| Zbyt wolny vs vLLM | **OK** — cel to edukacja, 50% wydajności vLLM = sukces |
| Safetensors parsing w C++ trudne | Ładuj wagi w Pythonie (PyTorch), przekaż raw pointery do C++ |

---

### Kluczowe papery

1. Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" (SOSP 2023)
2. Yu et al., "Orca" (OSDI 2022)
3. Dao et al., "FlashAttention" (NeurIPS 2022)
4. vLLM source (github.com/vllm-project/vllm)

**Dalsze pytania / decyzje do podjęcia:**

1. **Ile osób w zespole?** Plan zakłada 4, ale da się ściąć do 2-3 (łącząc role B+C lub A+B).
2. **Docelowy model**: wystarczy GPT-2 na obronę, czy Llama-2-7B jest wymagany?
3. **Zakres GQA**: czy obsługa Grouped Query Attention (Llama-2 style) jest w scope, czy wystarczy Multi-Head Attention (GPT-2 style)?