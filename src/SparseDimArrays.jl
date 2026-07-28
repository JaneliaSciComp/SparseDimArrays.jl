module SparseDimArrays

using Tables
using DimensionalData
import DimensionalData as DD

export sparsedimarray, sparsedimstack

# ---- shared index core: everything that depends only on the key columns,
# not on any particular value column, so multiple value columns coming from
# rows with the same keys (e.g. a mean and a sample-count derived from the
# same long table) can share one copy of it via `sparsedimstack` instead of
# each paying to load the key columns and build every Dict index again. ----
struct SparseDimIndex{N,D<:Tuple}
    dims::D
    keyvecs::NTuple{N,AbstractVector}
    key2pos::NTuple{N,Union{Nothing,Dict}}   # `nothing` for a `precoded` dimension
    postypes::NTuple{N,DataType}             # narrowest UInt type holding 1:length(dims[d])
    rowtype::DataType                        # narrowest UInt type holding 1:nrow(table)
    indices::Dict{Tuple{Vararg{Int}},Dict}
    # scalar lookups fix every dimension, so each full key maps to exactly one
    # row -- store that single row directly (Dict{fullkey=>row}) rather than a
    # Vector per key like `indices`, which for the full key would be one
    # 1-element Vector per row (all overhead). Lazily built; `nothing` until used.
    scalarindex::Base.RefValue{Any}
    lock::ReentrantLock
end

# narrowest unsigned integer type that can represent the values 1:n
function _postype(n::Integer)
    n <= typemax(UInt8)  ? UInt8  :
    n <= typemax(UInt16) ? UInt16 :
    n <= typemax(UInt32) ? UInt32 : UInt64
end

function _buildcore(table, keycols::NTuple{N,Symbol}, dims::Tuple,
                     precoded::NTuple{N,Bool}) where N
    Tables.istable(table) || throw(ArgumentError("`table` must be a Tables.jl-compatible source"))
    length(dims) == N || throw(ArgumentError("length(dims) ($(length(dims))) must equal length(keycols) ($N)"))

    cols = Tables.columns(table)
    fdims = DD.format(Tuple(dims))
    keyvecs = ntuple(i -> Tables.getcolumn(cols, keycols[i]), N)
    postypes = ntuple(i -> _postype(length(fdims[i])), N)
    rowtype = _postype(length(keyvecs[1]))
    key2pos = ntuple(N) do i
        precoded[i] ? nothing : Dict(v => postypes[i](p) for (p, v) in enumerate(DD.lookup(fdims[i])))
    end

    SparseDimIndex{N,typeof(fdims)}(fdims, keyvecs, key2pos, postypes, rowtype,
                                     Dict{Tuple{Vararg{Int}},Dict}(),
                                     Base.RefValue{Any}(nothing), ReentrantLock())
end

# position (within dimension d's lookup, as core.postypes[d]) of row r's key,
# whether or not that dimension's key column is already position-coded
@inline _pos(core::SparseDimIndex, d, r) =
    core.key2pos[d] === nothing ? core.postypes[d](core.keyvecs[d][r]) : core.key2pos[d][core.keyvecs[d][r]]

# the concrete (possibly heterogeneous, per-dimension-width) key type for the
# index over dimension-subset `subset`, e.g. subset=(1,3) -> Tuple{UInt16,UInt8}
_keytype(core::SparseDimIndex, subset::Tuple{Vararg{Int}}) =
    Tuple{ntuple(j -> core.postypes[subset[j]], length(subset))...}

# get-or-lazily-build-and-cache the full-key -> single-row index for scalar lookups
function _scalarindex!(core::SparseDimIndex{N}) where N
    lock(core.lock) do
        if core.scalarindex[] === nothing
            subset = ntuple(identity, N)
            KT, RT = _keytype(core, subset), core.rowtype
            idx = Dict{KT,RT}()
            for r in eachindex(core.keyvecs[1])
                idx[convert(KT, ntuple(d -> _pos(core, d, r), N))] = RT(r)
            end
            core.scalarindex[] = idx
        end
        core.scalarindex[]::Dict
    end
end

# get-or-lazily-build-and-cache the index for a subset of dimension numbers
# (e.g. subset=(1,3) -> keyed on (dim1 pos, dim3 pos)), row lists in core.rowtype
function _getindex!(core::SparseDimIndex, subset::Tuple{Vararg{Int}})
    lock(core.lock) do
        get!(core.indices, subset) do
            KT, RT = _keytype(core, subset), core.rowtype
            k = length(subset)
            idx = Dict{KT,Vector{RT}}()
            for r in eachindex(core.keyvecs[1])
                key = convert(KT, ntuple(j -> _pos(core, subset[j], r), k))
                push!(get!(() -> RT[], idx, key), RT(r))
            end
            idx
        end
    end
end

# ---- dimension-UNAWARE backing array. Deliberately a plain AbstractArray,
# with no knowledge of dims/selectors/At() at all: it's meant to be wrapped in
# a DimArray (single value column) or a DimStack (several, sharing one
# `core`), which already handle resolving At()/keyword selectors into plain
# positional indices and rebuilding the dimensioned result -- this type only
# has to answer plain `A[i,j,k]`-style indexing, exactly like a dense Array. ----
struct SparseArray{T,N,C<:SparseDimIndex} <: AbstractArray{T,N}
    core::C
    values::Vector{T}
    missingval::T
end
Base.size(A::SparseArray) = map(length, A.core.dims)

# scalar fast path: exact key tuple -> O(1) hash lookup, no intermediate array
function Base.getindex(A::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N}
    idx = _scalarindex!(A.core)
    row = get(idx, convert(_keytype(A.core, ntuple(identity, N)), I), nothing)
    isnothing(row) ? A.missingval : A.values[row]
end

_aslist(i::Integer, n) = (i,)
_aslist(i::AbstractVector{<:Integer}, n) = i
_aslist(::Base.Slice, n) = 1:n
_aslist(i::AbstractUnitRange{<:Integer}, n) = i
_isfull(i, n) = i isa Base.Slice || (i isa AbstractUnitRange && first(i) == 1 && last(i) == n)

const _StdIdx = Union{Integer,AbstractVector{<:Integer},Colon}

# every other index combination: builds a dense Array (with the same
# axis-dropping convention as ordinary Array indexing -- an Integer index
# drops that axis, everything else keeps it), so the DimArray/DimStack
# wrapping this type can rebuild its dims exactly as it would for any other
# backing array.
function Base.getindex(A::SparseArray{T,N}, I::Vararg{_StdIdx,N}) where {T,N}
    core = A.core
    ns = size(A)
    lists = ntuple(d -> _aslist(I[d], ns[d]), N)
    full  = ntuple(d -> _isfull(I[d], ns[d]), N)
    restricted = Tuple(d for d in 1:N if !full[d])
    # dimrole[d] === nothing for a "full" dim; otherwise its position within `restricted`
    dimrole = ntuple(d -> full[d] ? nothing : findfirst(==(d), restricted), N)

    out = fill(A.missingval, map(length, lists))

    if isempty(restricted)
        # nothing restricted at all: a full materialize, one pass over every row
        for r in eachindex(A.values)
            outidx = ntuple(d -> Int(_pos(core, d, r)), N)
            out[outidx...] = A.values[r]
        end
    else
        idx = _getindex!(core, restricted)
        KT = _keytype(core, restricted)
        selmap = ntuple(d -> full[d] ? nothing : Dict(v => i for (i, v) in enumerate(lists[d])), N)
        for combo in Iterators.product(ntuple(j -> lists[restricted[j]], length(restricted))...)
            rows = get(idx, convert(KT, combo), nothing)
            isnothing(rows) && continue
            restr_out = ntuple(j -> selmap[restricted[j]][combo[j]], length(restricted))
            for r in rows
                outidx = ntuple(d -> dimrole[d] === nothing ? Int(_pos(core, d, r)) : restr_out[dimrole[d]], N)
                out[outidx...] = A.values[r]
            end
        end
    end

    dropaxes = Tuple(d for d in 1:N if I[d] isa Integer)
    isempty(dropaxes) ? out : dropdims(out; dims=dropaxes)
end

"""
    sparsedimarray(table, keycols, valuecol, dims, missingval; indices=()) -> DimArray

Build a dense, N-dimensional `DimensionalData.DimArray` *view* over a
long/sparse [Tables.jl](https://github.com/JuliaData/Tables.jl) source: one
row per non-missing cell, `N` key columns (one per dimension) plus a value
column. Combinations absent from `table` read as `missingval`.

Indexing with `DimensionalData` selectors/keywords (`At`, `Near`, `dim=...`)
works exactly as for any other `DimArray` -- resolving a selector to a
dimension position, and rebuilding the sliced result, is handled entirely by
`DimensionalData`; this package only needs to answer, for a given tuple of
(position | vector of positions | `:`) per dimension, what values sit there.
That's answered by `Dict`-based indices keyed on whichever *subset* of
dimensions are fixed for a given call (e.g. a single dimension for
`A[dim1=At(x)]`, two for `A[dim1=At(x),dim2=At(y)]`), never a scan of the
full table. An index for a given dimension-subset is built once, lazily, on
first use, and cached -- or upfront via the `indices` keyword for subsets
known to be hot.

Every index packs its keys and row lists into the *narrowest* unsigned
integer type that fits (`UInt8`/`UInt16`/`UInt32`/`UInt64`, chosen per
dimension from its length, and once overall for row positions from
`length(table)`) -- this matters at the cardinalities a sparse table implies:
a `(dim1,dim2)` index can have as many groups as there are rows.

- `table`: any `Tables.istable` source (a `DataFrame`, `Arrow.Table`, `CSV.File`, ...)
- `keycols::NTuple{N,Symbol}`: column name holding the key for each dimension,
  in the same order as `dims`. Values need not be sorted, unique-per-row, or
  pre-mapped to integers -- any value present in the corresponding dimension's
  lookup is resolved to its position automatically.
- `valuecol::Symbol`: column read for each cell's value
- `dims::Tuple`: a `Dimension` per key column, e.g. `(Gene(genenames), Lineage(lineagenames))`
- `missingval::T`: value returned for key combinations absent from `table`
- `indices`: dimension-subsets (as `Tuple`s of 1-based dimension numbers,
  e.g. `((1,), (2,), (1,2))`) to build eagerly at construction time. Any other
  subset -- including `()`, a full materialize, and anything not listed here --
  is built (once) the first time a call actually needs it.
- `precoded::NTuple{N,Bool}`: mark a dimension `true` when its key column
  already holds the 1-based position within that dimension's lookup (e.g. a
  compact integer code column, stored instead of repeating a name string on
  every row) -- this skips building/using a `key2pos` map for that dimension
  entirely. Defaults to `false` for every dimension: the key column holds
  ordinary values (names, times, ...) resolved to a position via `key2pos`.

If several value columns come from rows sharing the same keys, use
[`sparsedimstack`](@ref) instead of calling this once per column -- it
builds the key->position maps and `Dict` indices only once and shares them.
"""
function sparsedimarray(table, keycols::NTuple{N,Symbol}, valuecol::Symbol,
                         dims::Tuple, missingval::T;
                         indices::Tuple = (), precoded::NTuple{N,Bool} = ntuple(_ -> false, N)) where {T,N}
    core = _buildcore(table, keycols, dims, precoded)
    foreach(subset -> _getindex!(core, subset), indices)
    vals = convert(Vector{T}, Tables.getcolumn(Tables.columns(table), valuecol))
    arr = SparseArray{T,N,typeof(core)}(core, vals, missingval)
    DimArray(arr, core.dims)
end

"""
    sparsedimstack(table, keycols, valuecols, dims, missingvals; indices=(), precoded=...) -> DimStack

Like [`sparsedimarray`](@ref), but for several value columns that share the
same key columns (e.g. a mean and a sample-count column derived from the same
underlying long table). Builds the key->position maps and `Dict` indices
*once*, and returns a `DimensionalData.DimStack` (layers named by
`valuecols`) whose layers all share them, rather than paying to load the key
columns and rebuild every index once per value column.

Each layer -- `stack.\$name` -- is a plain `DimArray`, indexable exactly like
one built by `sparsedimarray`. Indexing the *stack* itself, e.g.
`stack[dim1=At(x)]`, slices every layer together and returns another
`DimStack` (or a `NamedTuple`, if the selector is fully scalar) -- see
`DimensionalData.AbstractDimStack`.

`valuecols::NTuple{M,Symbol}` and `missingvals::Tuple` (one per value column,
possibly of different element types) replace `sparsedimarray`'s `valuecol`
and `missingval`; all other arguments are the same.
"""
function sparsedimstack(table, keycols::NTuple{N,Symbol}, valuecols::NTuple{M,Symbol},
                         dims::Tuple, missingvals::Tuple;
                         indices::Tuple = (), precoded::NTuple{N,Bool} = ntuple(_ -> false, N)) where {N,M}
    length(missingvals) == M || throw(ArgumentError("valuecols and missingvals must have the same length"))

    core = _buildcore(table, keycols, dims, precoded)
    foreach(subset -> _getindex!(core, subset), indices)

    cols = Tables.columns(table)
    arrays = ntuple(M) do i
        T = typeof(missingvals[i])
        vals = convert(Vector{T}, Tables.getcolumn(cols, valuecols[i]))
        SparseArray{T,N,typeof(core)}(core, vals, missingvals[i])
    end
    DimStack(NamedTuple{valuecols}(arrays), core.dims)
end

end # module SparseDimArrays
