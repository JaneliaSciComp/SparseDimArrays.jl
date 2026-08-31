module SparseDimArrays

using Tables
using DimensionalData
import DimensionalData as DD

export sparsedimarray, sparsedimstack

# ---- shared index core ----
# Rows are physically sorted once, at construction, lexicographically by their
# per-dimension positions. That sort order doubles as a zero-storage index for
# any *prefix* of the dimensions -- dim1, (dim1,dim2), ..., up to the full key --
# each answered by binary search (`searchsorted`) over a contiguous block, with
# cache-contiguous values. Non-prefix subsets (e.g. just dim2, or (dim1,dim3))
# aren't contiguous in that order, so they fall back to a lazily-built, cached
# hash `Dict`. Everything here depends only on the key columns, not on any value
# column, so several value columns sharing the same keys share one core via
# `sparsedimstack`.
struct SparseDimIndex{N,D<:Tuple}
    dims::D
    # per-row positions, PHYSICALLY SORTED lexicographically by (dim1,...,dimN);
    # both the row->position map and the sorted-prefix index.
    poscols::NTuple{N,Vector}
    postypes::NTuple{N,DataType}   # narrowest UInt type holding 1:length(dims[d])
    rowtype::DataType              # narrowest UInt type holding 1:nrow(table)
    indices::Dict{Tuple{Vararg{Int}},Dict}   # lazy hash indices for non-prefix subsets
    lock::ReentrantLock
end

# narrowest unsigned integer type that can represent the values 1:n
_postype(n::Integer) = n <= typemax(UInt8)  ? UInt8  :
                       n <= typemax(UInt16) ? UInt16 :
                       n <= typemax(UInt32) ? UInt32 : UInt64

# Object to read individual columns from with `Tables.getcolumn`, WITHOUT
# forcing every column of `table` to materialize. The Tables.jl contract only
# defines `getcolumn` on the object returned by `Tables.columns`, but some lazy
# columnar sources (e.g. Parquet2's `Dataset`) eagerly decode *every* column in
# their `Tables.columns` despite advertising `columnaccess`. Column-access
# tables define `getcolumn` directly on themselves (DataFrames, Arrow, CSV,
# Parquet2, NamedTuple column tables, ...), so use the table itself and only
# materialize genuinely row-oriented sources -- which have no per-column
# access to preserve anyway.
_colsource(table) = Tables.columnaccess(table) ? table : Tables.columns(table)

# Build the core and return it together with the sort permutation (original ->
# sorted row order), which the caller applies to each value column so values line
# up with the sorted `poscols` -- or `nothing` when the table's rows are already
# in sorted order, in which case the caller must skip the permutation.
function _buildcore(table, keycols::NTuple{N,Symbol}, dims::Tuple,
                     precoded::NTuple{N,Bool}) where N
    Tables.istable(table) || throw(ArgumentError("`table` must be a Tables.jl-compatible source"))
    length(dims) == N || throw(ArgumentError("length(dims) ($(length(dims))) must equal length(keycols) ($N)"))

    cols = _colsource(table)
    fdims = DD.format(Tuple(dims))
    keyvecs = ntuple(i -> Tables.getcolumn(cols, keycols[i]), N)
    postypes = ntuple(i -> _postype(length(fdims[i])), N)
    nrows = length(keyvecs[1])
    rowtype = _postype(nrows)
    key2pos = ntuple(N) do i
        precoded[i] ? nothing : Dict(v => postypes[i](p) for (p, v) in enumerate(DD.lookup(fdims[i])))
    end

    # position of every row in every dimension (unsorted) ...
    pos = ntuple(N) do d
        P, kv, k2p = postypes[d], keyvecs[d], key2pos[d]
        P[k2p === nothing ? P(kv[r]) : k2p[kv[r]] for r in 1:nrows]
    end
    # ... then sort rows lexicographically by (pos[1], ..., pos[N]). Sortedness
    # is by dimension POSITION, not raw key value -- the two coincide only when
    # each dim's lookup is itself in the keys' sort order -- so detect it from
    # `pos` directly (allocation-free) rather than trusting a caller flag, and
    # skip the sort and every downstream permutation copy when it already holds.
    if issorted(ntuple(d -> pos[d][r], N) for r in 1:nrows)
        perm = nothing
        poscols = pos
    else
        keytuples = [ntuple(d -> pos[d][r], N) for r in 1:nrows]
        perm = convert(Vector{rowtype}, sortperm(keytuples))
        poscols = ntuple(d -> pos[d][perm], N)
    end

    core = SparseDimIndex{N,typeof(fdims)}(fdims, poscols, postypes, rowtype,
                                            Dict{Tuple{Vararg{Int}},Dict}(), ReentrantLock())
    return core, perm
end

# Reorder a value column into the core's sorted row order. When the table was
# already sorted (`perm === nothing`) no copy is made at all -- `convert` is a
# no-op for a `Vector{T}` input -- so the result may ALIAS the table's own
# column. That sharing is deliberate: this package never mutates values, and it
# saves a full copy per value column. The flip side is that a caller who
# mutates the source table's column in place afterwards will see the array
# change with it.
_sortvals(::Type{T}, col, perm::Vector) where T = convert(Vector{T}, col)[perm]
_sortvals(::Type{T}, col, ::Nothing) where T = convert(Vector{T}, col)

# the concrete (possibly heterogeneous, per-dimension-width) key type for the
# index over dimension-subset `subset`, e.g. subset=(1,3) -> Tuple{UInt16,UInt8}
_keytype(core::SparseDimIndex, subset::Tuple{Vararg{Int}}) =
    Tuple{ntuple(j -> core.postypes[subset[j]], length(subset))...}

# Is `subset` a leading run (1, 2, ..., k)? Those are served by the sort order.
_isprefix(subset::Tuple{Vararg{Int}}) = subset === ntuple(identity, length(subset))

# Rows matching a prefix `combo`, as a contiguous range of (sorted) row indices:
# narrow [lo:hi] one dimension at a time with binary search. Within each block
# the next dimension's column is sorted (it's the primary key of that sub-block),
# so `searchsorted` is valid.
function _prefixrange(core::SparseDimIndex, subset::Tuple{Vararg{Int}}, combo::Tuple)
    lo = 1; hi = length(core.poscols[1])
    @inbounds for j in eachindex(subset)
        col = core.poscols[subset[j]]
        s = searchsorted(view(col, lo:hi), combo[j])
        isempty(s) && return 1:0
        hi = lo + last(s) - 1
        lo = lo + first(s) - 1
    end
    return lo:hi
end

# lazily build + cache the hash index for a NON-prefix subset; row lists are in
# sorted row order (matching `poscols` and the layers' reordered values)
function _getindex!(core::SparseDimIndex, subset::Tuple{Vararg{Int}})
    lock(core.lock) do
        get!(core.indices, subset) do
            KT, RT = _keytype(core, subset), core.rowtype
            k = length(subset)
            idx = Dict{KT,Vector{RT}}()
            for r in eachindex(core.poscols[1])
                key = convert(KT, ntuple(j -> core.poscols[subset[j]][r], k))
                push!(get!(() -> RT[], idx, key), RT(r))
            end
            idx
        end
    end
end

# ---- dimension-UNAWARE backing array. Deliberately a plain AbstractArray, with
# no knowledge of dims/selectors/At() at all: it's wrapped in a DimArray (single
# value column) or a DimStack (several, sharing one `core`), which handle
# resolving At()/keyword selectors into plain positional indices and rebuilding
# the dimensioned result -- this type only answers plain `A[i,j,k]`-style
# indexing, exactly like a dense Array. ----
struct SparseArray{T,N,C<:SparseDimIndex} <: AbstractArray{T,N}
    core::C
    values::Vector{T}   # in the core's sorted row order
    missingval::T
end
Base.size(A::SparseArray) = map(length, A.core.dims)

# scalar fast path: the full key is a prefix -> binary search to the single row
function Base.getindex(A::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N}
    rng = _prefixrange(A.core, ntuple(identity, N), I)
    isempty(rng) ? A.missingval : A.values[first(rng)]
end

_aslist(i::Integer, n) = (i,)
_aslist(i::AbstractVector{<:Integer}, n) = i
_aslist(::Base.Slice, n) = 1:n
_aslist(i::AbstractUnitRange{<:Integer}, n) = i
_isfull(i, n) = i isa Base.Slice || (i isa AbstractUnitRange && first(i) == 1 && last(i) == n)

const _StdIdx = Union{Integer,AbstractVector{<:Integer},Colon}

# every other index combination: builds a dense Array (with the same
# axis-dropping convention as ordinary Array indexing -- an Integer index drops
# that axis, everything else keeps it), so the DimArray/DimStack wrapping this
# type can rebuild its dims exactly as for any other backing array.
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
        # nothing restricted: full materialize, one pass over every row
        for r in eachindex(A.values)
            outidx = ntuple(d -> Int(core.poscols[d][r]), N)
            out[outidx...] = A.values[r]
        end
    else
        selmap = ntuple(d -> full[d] ? nothing : Dict(v => i for (i, v) in enumerate(lists[d])), N)
        KT = _keytype(core, restricted)
        prefix = _isprefix(restricted)
        idx = prefix ? nothing : _getindex!(core, restricted)   # searchsorted vs hash
        for combo in Iterators.product(ntuple(j -> lists[restricted[j]], length(restricted))...)
            ckey = convert(KT, combo)
            rows = prefix ? _prefixrange(core, restricted, ckey) : get(idx, ckey, nothing)
            (rows === nothing || isempty(rows)) && continue
            restr_out = ntuple(j -> selmap[restricted[j]][combo[j]], length(restricted))
            for r in rows
                outidx = ntuple(d -> dimrole[d] === nothing ? Int(core.poscols[d][r]) : restr_out[dimrole[d]], N)
                out[outidx...] = A.values[r]
            end
        end
    end

    dropaxes = Tuple(d for d in 1:N if I[d] isa Integer)
    isempty(dropaxes) ? out : dropdims(out; dims=dropaxes)
end

"""
    sparsedimarray(table, keycols, valuecol, dims, missingval; indices=(), precoded=...) -> DimArray

Build a dense, N-dimensional `DimensionalData.DimArray` *view* over a
long/sparse [Tables.jl](https://github.com/JuliaData/Tables.jl) source: one
row per non-missing cell, `N` key columns (one per dimension) plus a value
column. Combinations absent from `table` read as `missingval`.

Indexing with `DimensionalData` selectors/keywords (`At`, `Near`, `dim=...`)
works exactly as for any other `DimArray` -- resolving a selector to a
dimension position, and rebuilding the sliced result, is handled entirely by
`DimensionalData`; this package only answers, for a given tuple of (position |
vector of positions | `:`) per dimension, what values sit there.

Rows are sorted once (at construction) by their per-dimension positions, so a
query that fixes a *prefix* of the dimensions -- `dim1`, `(dim1,dim2)`, ..., or
the full key (a scalar lookup) -- is answered by binary search over a
contiguous block, at no extra storage. A query that fixes a non-prefix subset
(e.g. `dim2` alone, or `(dim1,dim3)`) uses a hash index built once, lazily, on
first use and cached -- or upfront via the `indices` keyword.

Every hash index packs its keys and row lists into the *narrowest* unsigned
integer type that fits (`UInt8`/`UInt16`/`UInt32`/`UInt64`, per dimension from
its length, and once overall for row positions from `length(table)`).

Only the key columns and the requested value column(s) are ever read from
`table`, so lazy columnar sources (e.g. a Parquet2 `Dataset`) never decode
unrelated columns. And if the table's rows already arrive sorted -- by
dimension *position*, which matches key-value order whenever each dimension's
lookup is itself sorted -- the sort is detected as unnecessary and skipped,
along with every permutation copy; the resulting array's value vector may then
share memory with the table's own column (nothing here ever mutates it, but an
in-place edit to the source column would show through).

- `table`: any `Tables.istable` source (a `DataFrame`, `Arrow.Table`, `CSV.File`, ...)
- `keycols::NTuple{N,Symbol}`: column name holding the key for each dimension,
  in the same order as `dims`. Values need not be sorted, unique-per-row, or
  pre-mapped to integers -- any value present in the corresponding dimension's
  lookup is resolved to its position automatically.
- `valuecol::Symbol`: column read for each cell's value
- `dims::Tuple`: a `Dimension` per key column, e.g. `(Gene(genenames), Lineage(lineagenames))`
- `missingval::T`: value returned for key combinations absent from `table`
- `indices`: non-prefix dimension-subsets (as `Tuple`s of 1-based dimension
  numbers, e.g. `((2,), (2,3))`) to build eagerly at construction. Prefix subsets
  are served by the sort order and need no hash index, so they are ignored here.
- `precoded::NTuple{N,Bool}`: mark a dimension `true` when its key column
  already holds the 1-based position within that dimension's lookup (e.g. a
  compact integer code column). Defaults to `false` for every dimension.

If several value columns come from rows sharing the same keys, use
[`sparsedimstack`](@ref) instead of calling this once per column -- it builds
the shared sort/index core only once.
"""
function sparsedimarray(table, keycols::NTuple{N,Symbol}, valuecol::Symbol,
                         dims::Tuple, missingval::T;
                         indices::Tuple = (), precoded::NTuple{N,Bool} = ntuple(_ -> false, N)) where {T,N}
    core, perm = _buildcore(table, keycols, dims, precoded)
    foreach(subset -> _isprefix(subset) || _getindex!(core, subset), indices)
    vals = _sortvals(T, Tables.getcolumn(_colsource(table), valuecol), perm)
    arr = SparseArray{T,N,typeof(core)}(core, vals, missingval)
    DimArray(arr, core.dims)
end

"""
    sparsedimstack(table, keycols, valuecols, dims, missingvals; indices=(), precoded=...) -> DimStack

Like [`sparsedimarray`](@ref), but for several value columns that share the
same key columns (e.g. a mean and a sample-count column derived from the same
underlying long table). Builds the shared sort/index core *once*, and returns a
`DimensionalData.DimStack` (layers named by `valuecols`) whose layers all share
it, rather than paying to sort and index once per value column.

Each layer -- `stack.\$name` -- is a plain `DimArray`, indexable exactly like
one built by `sparsedimarray`. Indexing the *stack* itself, e.g.
`stack[dim1=At(x)]`, slices every layer together and returns another `DimStack`
(or a `NamedTuple`, if the selector is fully scalar).

`valuecols::NTuple{M,Symbol}` and `missingvals::Tuple` (one per value column,
possibly of different element types) replace `sparsedimarray`'s `valuecol`
and `missingval`; all other arguments are the same.
"""
function sparsedimstack(table, keycols::NTuple{N,Symbol}, valuecols::NTuple{M,Symbol},
                         dims::Tuple, missingvals::Tuple;
                         indices::Tuple = (), precoded::NTuple{N,Bool} = ntuple(_ -> false, N)) where {N,M}
    length(missingvals) == M || throw(ArgumentError("valuecols and missingvals must have the same length"))

    core, perm = _buildcore(table, keycols, dims, precoded)
    foreach(subset -> _isprefix(subset) || _getindex!(core, subset), indices)

    cols = _colsource(table)
    arrays = ntuple(M) do i
        T = typeof(missingvals[i])
        vals = _sortvals(T, Tables.getcolumn(cols, valuecols[i]), perm)
        SparseArray{T,N,typeof(core)}(core, vals, missingvals[i])
    end
    DimStack(NamedTuple{valuecols}(arrays), core.dims)
end

end # module SparseDimArrays
