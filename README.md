# SparseDimArrays.jl

`SparseDimArrays.jl` wraps a long/sparse [Tables.jl](https://github.com/JuliaData/Tables.jl)
source -- one row per non-missing cell, N key columns plus a value column --
as a dense, N-dimensional `AbstractArray` with named-dimension indexing, via
[DimensionalData.jl](https://github.com/rafaqz/DimensionalData.jl). Key
combinations absent from the table read as a caller-supplied `missingval`,
without ever densifying the table into an actual N-dimensional cube in
memory.

## Why

Given a table with several key columns and a value column, you can already do
this by hand with `DataFrames.groupby`. This package exists because:

- **Fast along *any* subset of dimensions, not just one.** `A[dim1=At(x)]`,
  `A[dim2=At(y)]`, and `A[dim1=At(x),dim2=At(y)]` are each answered by a
  `Dict` built (once, lazily) from exactly the dimensions that call fixed --
  never a scan of the full table, and never built for combinations that are
  never asked for.
- **Not tied to `DataFrames.jl`.** The only dependency is `Tables.jl`, so a
  `DataFrame`, an `Arrow.Table`, a `CSV.File`, or a bare `NamedTuple` of
  vectors all work the same way, without pulling in all of DataFrames.jl.
- **Memory-conscious at sparse-table cardinality.** At high sparsity, an
  index over two dimensions can have nearly as many groups as the table has
  rows. Every index packs its keys and row-position lists into the
  narrowest unsigned integer type that fits (`UInt8`/`UInt16`/`UInt32`),
  rather than machine-width `Int`.
- **Plays directly with `DimensionalData.jl`.** Once constructed, a
  `SparseDimArray` *is* a `DimensionalData.AbstractBasicDimArray` -- `At`,
  `Near`, keyword indexing, `set`, `cat`, further slicing, all just work,
  and every non-scalar read returns a real, dense `DimArray`.

We didn't find an existing package that does this; see the design notes
below for what was considered and why.

## Usage

```julia
using SparseDimArrays, DimensionalData, DataFrames

const Sensor = Dim{:Sensor}
const Site   = Dim{:Site}
const Day    = Dim{:Day}

# one row per (sensor, site, day) actually observed
table = DataFrame(sensor=["s1","s1","s2"],
                   site=["siteA","siteB","siteA"],
                   day=Int16[1,1,2],
                   reading=Float32[12.3, 0.0, 4.1])

sensors = unique(table.sensor)
sites   = unique(table.site)
days    = sort(unique(table.day))

A = SparseDimArray(table, (:sensor, :site, :day), :reading,
                   (Sensor(sensors), Site(sites), Day(days)), NaN32)

A[Sensor=At("s1")]                       # SitexDay DimArray, NaN where absent
A[Site=At("siteA"), Sensor=At("s1")]      # Day DimVector
A[Sensor=At("s1"), Site=At("siteB"), Day=At(Int16(1))]  # scalar
```

### Compact integer-coded columns

If a key column already stores a compact 1-based integer code (rather than
repeating a name string on every row -- exactly how large sparse tables
should be stored), mark that dimension `precoded` and skip the value->position
mapping for it entirely:

```julia
table2 = DataFrame(isensor=[1,1,2], isite=[1,2,1], day=Int16[1,1,2],
                    reading=Float32[12.3, 0.0, 4.1])   # isensor/isite are 1-based
                                                        # positions into sensors/sites
A2 = SparseDimArray(table2, (:isensor, :isite, :day), :reading,
                    (Sensor(sensors), Site(sites), Day(days)), NaN32;
                    precoded=(true, true, false))
```

`precoded` only changes how a *row's* key-column value maps to a position
internally (directly, vs. via a `value -> position` `Dict` built from the
dimension's lookup). It has no effect on how you *query* the array: `At(x)`
is always resolved against the dimension's lookup -- `Sensor(sensors)` above
-- regardless of `precoded`, so `A2[Sensor=At("s1")]` still works exactly as
it does for `A`, even though the table underneath `A2` never stores the
string `"s1"` at all.

### Eager vs. lazy indices

By default, the `Dict` index for a given dimension-subset (e.g. dimensions
`(1,)`, or `(1,2)`) is built the first time a call needs it, then cached. Pass
`indices` to build specific subsets upfront instead, e.g. if you know which
access patterns will be hot:

```julia
A = SparseDimArray(table, keycols, valuecol, dims, missingval;
                   indices=((1,), (1,2)))
```

## Design notes / prior art considered

`AxisKeys.jl` and `NamedDims.jl` attach names/key-vectors to an
already-materialized dense array -- they don't provide a sparse-table-backed
storage layer, and `AxisKeys`'s own key lookup is a linear scan by design.
`IndexedTables.jl`'s `NDSparse` is the closest conceptual match, but it's
sorted by one fixed key order (fast along that prefix only; multiple access
patterns need multiple physically-resorted copies), has no
`DimensionalData.jl` integration, and its host package (`JuliaDB.jl`) is
explicitly unmaintained. `SparseArrayKit.jl` is a `CartesianIndex`-keyed DOK
sparse array for tensor algebra, with no named dimensions or `Tables.jl`
input. `DimensionalData.jl` itself has a `DimArray(table, dims)` constructor,
but it *eagerly* densifies the table via `restore_array` -- the opposite of
what a very sparse, multi-gigabyte table needs.

## License

BSD 3-Clause, see [LICENSE.txt](LICENSE.txt).
