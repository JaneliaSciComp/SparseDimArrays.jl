module SparseDimArrays

using Tables
using DimensionalData
import DimensionalData as DD
import DimensionalData: AbstractBasicDimArray

export SparseDimArray

"""
    SparseDimArray{T,N,D} <: DimensionalData.AbstractBasicDimArray{T,N,D}

A dense, N-dimensional array *view* over a long/sparse [Tables.jl](https://github.com/JuliaData/Tables.jl)
source: one row per non-missing cell, `N` key columns (one per dimension) plus
a value column. Combinations absent from `table` read as `missingval`.

Indexing with `DimensionalData` selectors/keywords (`At`, `Near`, `dim=...`)
works exactly as for a dense `DimArray` -- resolving a selector to a
dimension position is handled entirely by `DimensionalData`; this package
only needs to answer, for a given tuple of (position | vector of positions |
`:`) per dimension, what values sit there. That's answered by `Dict`-based
indices keyed on whichever *subset* of dimensions are fixed for a given call
(e.g. a single dimension for `A[dim1=At(x)]`, two for `A[dim1=At(x),dim2=At(y)]`),
never a scan of the full table. An index for a given dimension-subset is
built once, lazily, on first use, and cached -- or upfront via the `indices`
keyword for subsets known to be hot.

    SparseDimArray(table, keycols, valuecol, dims, missingval; indices=())

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
"""
struct SparseDimArray{T,N,D<:Tuple} <: AbstractBasicDimArray{T,N,D}
    keyvecs::NTuple{N,AbstractVector}
    values::Vector{T}
    missingval::T
    dims::D
    key2pos::NTuple{N,Union{Nothing,Dict}}   # `nothing` for a `precoded` dimension
    indices::Dict{Tuple{Vararg{Int}},Dict}
    lock::ReentrantLock
end

DD.dims(A::SparseDimArray) = A.dims
Base.size(A::SparseDimArray) = map(length, A.dims)

# position (within dimension d's lookup) of row r's key, whether or not that
# dimension's key column is already position-coded
@inline _pos(A::SparseDimArray, d, r) =
    A.key2pos[d] === nothing ? Int(A.keyvecs[d][r]) : A.key2pos[d][A.keyvecs[d][r]]

function SparseDimArray(table, keycols::NTuple{N,Symbol}, valuecol::Symbol,
                         dims::Tuple, missingval::T;
                         indices::Tuple = (), precoded::NTuple{N,Bool} = ntuple(_ -> false, N)) where {T,N}
    Tables.istable(table) || throw(ArgumentError("`table` must be a Tables.jl-compatible source"))
    length(dims) == N || throw(ArgumentError("length(dims) ($(length(dims))) must equal length(keycols) ($N)"))

    cols = Tables.columns(table)
    fdims = DD.format(Tuple(dims))
    keyvecs = ntuple(i -> Tables.getcolumn(cols, keycols[i]), N)
    vals = convert(Vector{T}, Tables.getcolumn(cols, valuecol))
    key2pos = ntuple(N) do i
        precoded[i] ? nothing : Dict(v => p for (p, v) in enumerate(DD.lookup(fdims[i])))
    end

    A = SparseDimArray{T,N,typeof(fdims)}(keyvecs, vals, missingval, fdims, key2pos,
                                          Dict{Tuple{Vararg{Int}},Dict}(), ReentrantLock())
    foreach(subset -> _getindex!(A, subset), indices)
    return A
end

# get-or-lazily-build-and-cache the Dict{NTuple{k,Int},Vector{Int32}} index for
# a subset of dimension numbers, e.g. subset=(1,3) -> keyed on (dim1 pos, dim3 pos)
function _getindex!(A::SparseDimArray, subset::Tuple{Vararg{Int}})
    lock(A.lock) do
        get!(A.indices, subset) do
            k = length(subset)
            idx = Dict{NTuple{k,Int},Vector{Int32}}()
            for r in eachindex(A.values)
                key = ntuple(j -> _pos(A, subset[j], r), k)
                push!(get!(() -> Int32[], idx, key), Int32(r))
            end
            idx
        end
    end
end

# scalar fast path: exact key tuple -> O(1) hash lookup, no intermediate array
function Base.getindex(A::SparseDimArray{T,N}, I::Vararg{Int,N}) where {T,N}
    subset = ntuple(identity, N)
    idx = _getindex!(A, subset)
    rows = get(idx, I, nothing)
    (isnothing(rows) || isempty(rows)) ? A.missingval : A.values[first(rows)]
end

_aslist(i::Integer, n) = (i,)
_aslist(i::AbstractVector{<:Integer}, n) = i
_aslist(::Base.Slice, n) = 1:n
_aslist(i::AbstractUnitRange{<:Integer}, n) = i
_isfull(i, n) = i isa Base.Slice || (i isa AbstractUnitRange && first(i) == 1 && last(i) == n)

# every other index combination: always materializes a real, dense DimArray,
# exactly what DimensionalData would build for a plain backing Array.
function DD.rebuildsliced(f::Function, A::SparseDimArray{T,N}, I::Tuple, name=DD.name(A)) where {T,N}
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
            outidx = ntuple(d -> _pos(A, d, r), N)
            out[outidx...] = A.values[r]
        end
    else
        idx = _getindex!(A, restricted)
        selmap = ntuple(d -> full[d] ? nothing : Dict(v => i for (i, v) in enumerate(lists[d])), N)
        for combo in Iterators.product(ntuple(j -> lists[restricted[j]], length(restricted))...)
            rows = get(idx, combo, nothing)
            isnothing(rows) && continue
            restr_out = ntuple(j -> selmap[restricted[j]][combo[j]], length(restricted))
            for r in rows
                outidx = ntuple(d -> dimrole[d] === nothing ? _pos(A, d, r) : restr_out[dimrole[d]], N)
                out[outidx...] = A.values[r]
            end
        end
    end

    dropaxes = Tuple(d for d in 1:N if I[d] isa Integer)
    out2 = isempty(dropaxes) ? out : dropdims(out; dims=dropaxes)
    newdims, newrefdims = DD.slicedims(f, A, I)
    return DimArray(out2, newdims; refdims=newrefdims, name)
end

end # module SparseDimArrays
