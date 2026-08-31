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
# [`GroupIndex`](@ref). Everything here depends only on the key columns, not on
# any value column, so several value columns sharing the same keys share one
# core via `sparsedimstack`.
struct SparseDimIndex{N,D<:Tuple,PC<:NTuple{N,Vector},R<:Unsigned}
    dims::D
    # per-row positions, PHYSICALLY SORTED lexicographically by (dim1,...,dimN);
    # both the row->position map and the sorted-prefix index. Concretely typed
    # in PC (per dimension, the narrowest UInt holding 1:length(dims[d])) so
    # the per-row query loops specialize instead of boxing on every access.
    poscols::PC
    indices::Dict{Tuple{Vararg{Int}},Any}   # lazy GroupIndex per non-prefix subset
    lock::ReentrantLock
end

# derived, not stored: the narrow per-dimension position types live in PC's
# element types, and the narrow row-id type (holding 1:nrow(table)) in R
_postypes(core::SparseDimIndex) = map(eltype, core.poscols)
_rowtype(::SparseDimIndex{N,D,PC,R}) where {N,D,PC,R} = R

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

# Function barriers for the per-row construction loops. The narrow position
# types are picked at runtime (`postypes`/`rowtype` are `DataType`s, as the
# struct stores them), so loops touching them directly are type-unstable and
# pay a boxing dynamic dispatch on EVERY row; routing through these barriers
# specializes each loop on the concrete types instead -- one dynamic call per
# dimension, not one per row.
_key2pos(::Type{P}, lookup) where P = Dict(v => P(p) for (p, v) in enumerate(lookup))
# A precoded key column already stored at the narrow position width is reused
# as-is, not copied -- when the rows turn out presorted the core's poscols then
# alias the table's own column (the unsorted path copies via `pos[d][perm]`
# regardless). Same deliberate read-only sharing as `_sortvals` on values.
_poscol(::Type{P}, kv::Vector{P}, ::Nothing) where P = kv
_poscol(::Type{P}, kv, ::Nothing) where P = P[P(v) for v in kv]
_poscol(::Type{P}, kv, k2p::Dict) where P = P[k2p[v] for v in kv]

# Sortedness check + sort + permute (also behind the barrier). Sortedness is by
# dimension POSITION, not raw key value -- the two coincide only when each
# dim's lookup is itself in the keys' sort order -- so detect it from `pos`
# directly (allocation-free) rather than trusting a caller flag, and skip the
# sort and every downstream permutation copy (`perm === nothing`) when it
# already holds.
function _sortpos(pos::NTuple{N,AbstractVector}, ::Type{R}) where {N,R}
    nrows = length(pos[1])
    issorted(ntuple(d -> pos[d][r], N) for r in 1:nrows) && return nothing, pos
    keytuples = [ntuple(d -> pos[d][r], N) for r in 1:nrows]
    perm = convert(Vector{R}, sortperm(keytuples))
    return perm, ntuple(d -> pos[d][perm], N)
end

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
        precoded[i] ? nothing : _key2pos(postypes[i], DD.lookup(fdims[i]))
    end

    # position of every row in every dimension (unsorted), then sort rows
    # lexicographically by (pos[1], ..., pos[N]) -- unless they already are
    pos = ntuple(d -> _poscol(postypes[d], keyvecs[d], key2pos[d]), N)
    perm, poscols = _sortpos(pos, rowtype)

    core = SparseDimIndex{N,typeof(fdims),typeof(poscols),rowtype}(
        fdims, poscols, Dict{Tuple{Vararg{Int}},Any}(), ReentrantLock())
    return core, perm
end

# Reorder a value column into the core's sorted row order, gathering (and
# eltype-converting, if needed) in a single pass -- one allocation even when
# `col` is not already a `Vector{T}`. When the table was already sorted
# (`perm === nothing`) no copy is made at all -- `convert` is a no-op for a
# `Vector{T}` input -- so the result may ALIAS the table's own
# column. That sharing is deliberate: this package never mutates values, and it
# saves a full copy per value column. The flip side is that a caller who
# mutates the source table's column in place afterwards will see the array
# change with it.
_sortvals(::Type{T}, col, perm::Vector) where T = T[col[p] for p in perm]
_sortvals(::Type{T}, col, ::Nothing) where T = convert(Vector{T}, col)

# CSR-style index over one non-prefix dimension subset: every row id, grouped
# by that subset's (narrow, possibly heterogeneous-width) key, in ONE
# exact-length vector, plus key -> block range. Compared to the obvious
# Dict{key,Vector{row}} this avoids a per-key Vector header and push!-growth
# capacity slop -- which together more than doubled resident size for
# many-key subsets -- and makes each group's rows contiguous.
struct GroupIndex{KT,RT}
    rows::Vector{RT}                # all row ids; ascending within each group
    ranges::Dict{KT,UnitRange{RT}}  # key -> block in `rows`
end
Base.length(g::GroupIndex) = length(g.ranges)

# Two passes over the subset's position columns (specialized on their concrete
# types via the outer dynamic call): count rows per key, carve `rows` into one
# exclusive block per key, then fill -- every group exactly sized, no push!.
_groupindex(cols::Tuple{Vararg{Vector}}, ::Type{RT}) where RT =
    _groupindex(Tuple{map(eltype, cols)...}, cols, RT)
function _groupindex(::Type{KT}, cols::NTuple{K,Vector}, ::Type{RT}) where {KT,K,RT}
    n = length(cols[1])
    counts = Dict{KT,Int}()
    for r in 1:n
        key = map(c -> @inbounds(c[r]), cols)
        counts[key] = get(counts, key, 0) + 1
    end
    ranges = Dict{KT,UnitRange{RT}}()
    sizehint!(ranges, length(counts))
    stop = 0
    for (key, c) in counts
        ranges[key] = RT(stop+1):RT(stop+c)
        stop += c
    end
    for (key, rng) in ranges   # reuse `counts` as each key's next-write cursor
        counts[key] = Int(first(rng))
    end
    rows = Vector{RT}(undef, n)
    for r in 1:n
        key = map(c -> @inbounds(c[r]), cols)
        i = counts[key]
        @inbounds rows[i] = RT(r)
        counts[key] = i + 1
    end
    GroupIndex{KT,RT}(rows, ranges)
end

# rows matching `combo` in this index, as a contiguous view -- or `nothing`
function _grouprows(g::GroupIndex{KT}, combo::Tuple) where KT
    rng = get(g.ranges, convert(KT, combo), nothing)
    rng === nothing ? nothing : view(g.rows, Int(first(rng)):Int(last(rng)))
end

# Is `subset` a leading run (1, 2, ..., k)? Those are served by the sort order.
_isprefix(subset::Tuple{Vararg{Int}}) = subset === ntuple(identity, length(subset))

# Rows matching a prefix `combo`, as a contiguous range of (sorted) row indices:
# narrow [lo:hi] one dimension at a time with binary search. Within each block
# the next dimension's column is sorted (it's the primary key of that sub-block),
# so `searchsorted` is valid. Recurse down the poscols tuple (rather than loop
# with a runtime index) so each level's column is CONCRETELY typed -- a
# union-typed column would heap-allocate its `view` on every lookup.
_prefixrange(core::SparseDimIndex, combo::Tuple) =
    _prefixwalk(core.poscols, combo, 1, length(core.poscols[1]))
_prefixwalk(cols::Tuple, ::Tuple{}, lo::Int, hi::Int) = lo:hi
function _prefixwalk(cols::Tuple, combo::Tuple, lo::Int, hi::Int)
    s = searchsorted(view(cols[1], lo:hi), combo[1])
    isempty(s) && return 1:0
    _prefixwalk(Base.tail(cols), Base.tail(combo), lo + first(s) - 1, lo + last(s) - 1)
end

# lazily build + cache the group index for a NON-prefix subset; row ids are in
# sorted row order (matching `poscols` and the layers' reordered values)
function _getindex!(core::SparseDimIndex, subset::Tuple{Vararg{Int}})
    lock(core.lock) do
        get!(core.indices, subset) do
            _groupindex(map(i -> core.poscols[i], subset), _rowtype(core))
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
    rng = _prefixrange(A.core, I)
    isempty(rng) ? A.missingval : A.values[first(rng)]
end

_aslist(i::Integer, n) = (i,)
_aslist(i::AbstractVector{<:Integer}, n) = i
_aslist(::Base.Slice, n) = 1:n
_aslist(i::AbstractUnitRange{<:Integer}, n) = i
_isfull(i, n) = i isa Base.Slice || (i isa AbstractUnitRange && first(i) == 1 && last(i) == n)

const _StdIdx = Union{Integer,AbstractVector{<:Integer},Colon}

# The full-materialize and per-combo row-write loops, behind function barriers
# so they specialize on the concrete narrow poscols types: one dynamic call per
# query (or per key-combo) instead of boxing on every row.
function _fillall!(out::Array{T,N}, values::Vector{T}, poscols::NTuple{N,Vector}) where {T,N}
    @inbounds for r in eachindex(values)
        out[ntuple(d -> Int(poscols[d][r]), Val(N))...] = values[r]
    end
end
# `dimrole[d]` is the output index source for dimension d: `nothing` -> the
# row's own position, an integer -> that slot of the (per-combo) `restr_out`.
# `map` over the tuples (not `ntuple` over 1:N) so each element call sees the
# CONCRETE column/role types -- runtime tuple indexing here boxes every access.
function _writerows!(out::Array{T,N}, values::Vector{T}, poscols::NTuple{N,Vector},
                      rows, dimrole::NTuple{N,Union{Nothing,Int}}, restr_out) where {T,N}
    @inbounds for r in rows
        outidx = map((c, role) -> role === nothing ? Int(c[r]) : Int(restr_out[role]),
                     poscols, dimrole)
        out[outidx...] = values[r]
    end
end

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
        _fillall!(out, A.values, core.poscols)
    else
        selmap = ntuple(d -> full[d] ? nothing : Dict(v => i for (i, v) in enumerate(lists[d])), N)
        prefix = _isprefix(restricted)
        idx = prefix ? nothing : _getindex!(core, restricted)   # searchsorted vs group index
        for combo in Iterators.product(ntuple(j -> lists[restricted[j]], length(restricted))...)
            rows = prefix ? _prefixrange(core, combo) : _grouprows(idx, combo)
            (rows === nothing || isempty(rows)) && continue
            restr_out = ntuple(j -> selmap[restricted[j]][combo[j]], length(restricted))
            _writerows!(out, A.values, core.poscols, rows, dimrole, restr_out)
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
(e.g. `dim2` alone, or `(dim1,dim3)`) uses a group index built once, lazily, on
first use and cached -- or upfront via the `indices` keyword.

Every group index stores all row ids, grouped by key, in one exact-length
vector plus a key -> block-range table (no per-key vectors to over-allocate),
packed into the *narrowest* unsigned integer type that fits
(`UInt8`/`UInt16`/`UInt32`/`UInt64`, per dimension from its length, and once
overall for row ids from `length(table)`).

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
