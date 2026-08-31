# SparseDimArrays.jl release notes

## v0.2.0

Construction and query memory overhauled. The public API (`sparsedimarray`,
`sparsedimstack`) is unchanged; all numbers below are from the bundled
benchmark (500k-row table, 1000 x 100 x 50 dims, 10% filled).

### Queries: allocation-free lookups, output-only slices

The index core is now concretely typed (narrow position/row-id types are type
parameters, not runtime fields), and every per-row query loop specializes on
those types instead of boxing:

- scalar lookup `A[i,j,k]`: 224 bytes / 1.1 us -> **0 bytes / 250 ns**
- single-dimension slice: 2.2 MiB / 3 ms -> **205 KiB / 13 us** (the
  remaining allocation is the returned array itself)
- full materialize `A[:,:,:]`: 45.7 MiB / 198 ms -> **19.1 MiB / 5 ms**
  (19.1 MiB is the dense output)

### Half the resident index memory

Non-prefix dimension subsets are now served by a CSR-style `GroupIndex` -- all
row ids grouped by key in one exact-length vector plus a key -> block-range
table -- instead of a `Dict` of per-key vectors, whose headers and `push!`
growth slop more than doubled resident size. Shared core on the benchmark:
24.9 MiB -> **11.3 MiB**; every index now sits within a few percent of its
theoretical minimum.

### Construction: read less, copy less, sort only when needed

- Only the key columns and requested value columns are read from the table,
  each exactly once. Lazy columnar sources (e.g. a Parquet2 `Dataset`) no
  longer decode unrelated columns -- previously the entire table was decoded
  twice.
- If rows already arrive sorted by dimension position (detected automatically,
  allocation-free), the sort and every permutation copy are skipped:
  construction from a presorted table costs 3.4 ms / 979 KiB vs 56 ms /
  16.2 MiB shuffled.
- Precoded key columns already stored at the narrow position width are reused
  as-is; presorted + precoded construction copies nothing at all.
- The per-row position-encoding loops no longer box (1.5M allocations
  eliminated): shuffled construction dropped 146 ms / 40.2 MiB -> 56 ms /
  16.2 MiB.
- Value columns are gathered and eltype-converted in a single pass.

### Behavior and compatibility notes

- **Aliasing (deliberate):** when rows arrive presorted, the array's value
  vector -- and any already-narrow precoded position column -- may share
  memory with the table's own column rather than being copied. The package
  never mutates these, but an in-place edit to the source column will show
  through to the array.
- Internals changed shape: `SparseDimIndex` no longer has `postypes`/`rowtype`
  fields (use the type parameters), and `core.indices` holds `GroupIndex`
  values instead of `Dict`s. Code touching only the public API is unaffected.

### Benchmarks

The `construct` suite now covers {shuffled, presorted} x {in-memory
DataFrame, lazy Parquet2 `Dataset`}, with the parquet files padded by
never-read columns to verify selective decoding.

## v0.1.0

Initial release: `sparsedimarray` and `sparsedimstack` build dense
`DimensionalData` views over long/sparse Tables.jl sources, with rows sorted
once so prefix queries are answered by binary search and other subsets by
lazily built, narrowly typed hash indices.
