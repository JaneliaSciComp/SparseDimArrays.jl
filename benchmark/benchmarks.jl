# Benchmark suite for SparseDimArrays.
#
# Run standalone:
#     julia --project=benchmark benchmark/benchmarks.jl
# or with PkgBenchmark (uses the `SUITE` defined here):
#     using PkgBenchmark; benchmarkpkg("SparseDimArrays")
#
# The suite covers construction -- {shuffled, presorted} rows crossed with
# {in-memory DataFrame, lazy Parquet2 Dataset} sources -- and every access
# pattern the design cares about:
# the scalar (full-key) lookup, single-dimension slices, a pair slice, a full
# materialize, and the *cold* cost of building an index on first use (the part
# the sorted-prefix work changes).

using BenchmarkTools, SparseDimArrays, DimensionalData, DataFrames, Parquet2, Random, Printf, PrettyTables

const Gene = Dim{:Gene}
const Lineage = Dim{:Lineage}
const Time = Dim{:Time}

# A synthetic sparse table in the integer-coded ("precoded") layout the package
# targets: iGE/iLI are 1-based positions, TI an Int16 timepoint. `fill_frac` of
# the dense GExLIxTI cube is present.
function make_table(; ngenes=1000, nli=100, nti=50, fill_frac=0.1, seed=1)
    rng = MersenneTwister(seed)
    genes     = ["g$i" for i in 1:ngenes]
    lineages  = ["l$i" for i in 1:nli]
    times     = Int16.(1:nti)
    ncell = ngenes * nli * nti
    nrow  = round(Int, fill_frac * ncell)
    # sample distinct (gene,lineage,time) cells
    cells = randperm(rng, ncell)[1:nrow]
    iGE = Vector{UInt16}(undef, nrow); iLI = Vector{UInt16}(undef, nrow)
    TI  = Vector{Int16}(undef, nrow);  val = rand(rng, Float32, nrow)
    for (r, c) in enumerate(cells)
        c0 = c - 1
        iGE[r] = UInt16(c0 % ngenes + 1)
        iLI[r] = UInt16((c0 ÷ ngenes) % nli + 1)
        TI[r]  = Int16((c0 ÷ (ngenes*nli)) + 1)
    end
    tab = DataFrame(iGE=iGE, iLI=iLI, TI=TI, val=val)
    (; tab, genes, lineages, times, ngenes, nli, nti)
end

const D = make_table()
const dims = (Gene(D.genes), Lineage(D.lineages), Time(D.times))

# the same rows presorted by position: construction detects this and skips the
# sort and every permutation copy (values then alias the table's column)
const sortedtab = sort(D.tab, [:iGE, :iLI, :TI])

# The same tables as lazy Parquet2 `Dataset`s, padded with `npad` value columns
# that are never asked for: construction reads columns selectively, so the
# padding must never decode. (An in-memory DataFrame hands back its columns for
# free either way -- lazy sources are where selective reads matter.)
function parquet_dataset(tab; npad=4)
    path = joinpath(mktempdir(), "bench.parquet")
    padded = copy(tab)
    rng = MersenneTwister(0)
    for i in 1:npad
        padded[!, "pad$i"] = rand(rng, Float32, nrow(tab))
    end
    Parquet2.writefile(path, padded)
    Parquet2.Dataset(path)
end
const pq       = parquet_dataset(D.tab)
const pqsorted = parquet_dataset(sortedtab)

mkarray(tab=D.tab) = sparsedimarray(tab, (:iGE, :iLI, :TI), :val, dims, NaN32;
                                    precoded=(true, true, false))

const SUITE = BenchmarkGroup()

# --- construction: {shuffled, presorted} x {in-memory DataFrame, lazy Parquet2} ---
SUITE["construct"] = BenchmarkGroup()
SUITE["construct"]["array"]             = @benchmarkable mkarray()
SUITE["construct"]["array-presorted"]   = @benchmarkable mkarray(sortedtab)
SUITE["construct"]["parquet"]           = @benchmarkable mkarray(pq)
SUITE["construct"]["parquet-presorted"] = @benchmarkable mkarray(pqsorted)
SUITE["construct"]["stack"] = @benchmarkable sparsedimstack(D.tab, (:iGE, :iLI, :TI),
    (:val,), dims, (NaN32,); precoded=(true, true, false))

# --- warm queries (indices already built; BenchmarkTools' minimum excludes the
#     one-time lazy build on the first sample) ---
const A = mkarray()
A[Gene=At(D.genes[1])]; A[Lineage=At(D.lineages[1])]; A[Time=At(D.times[1])]  # warm indices
A[Lineage=At(D.lineages[1]), Gene=At(D.genes[1])]; A[1, 1, 1]

SUITE["warm"] = BenchmarkGroup()
SUITE["warm"]["scalar"] = @benchmarkable A[i, j, k] setup=(
    i=rand(1:D.ngenes); j=rand(1:D.nli); k=rand(1:D.nti))
SUITE["warm"]["gene"]     = @benchmarkable A[Gene=At(g)]     setup=(g=rand(D.genes))
SUITE["warm"]["lineage"]  = @benchmarkable A[Lineage=At(l)]  setup=(l=rand(D.lineages))
SUITE["warm"]["time"]     = @benchmarkable A[Time=At(t)]     setup=(t=rand(D.times))
SUITE["warm"]["gene+lineage"] = @benchmarkable A[Lineage=At(l), Gene=At(g)] setup=(
    g=rand(D.genes); l=rand(D.lineages))
SUITE["warm"]["materialize"] = @benchmarkable A[:, :, :]

# --- cold index build: fresh array (no indices) then one slice, so the timed
#     work includes building that subset's index from scratch ---
SUITE["cold"] = BenchmarkGroup()
SUITE["cold"]["gene"]   = @benchmarkable Af[Gene=At(g)] setup=(Af=mkarray(); g=rand(D.genes))
SUITE["cold"]["scalar"] = @benchmarkable Af[i, j, k]    setup=(
    Af=mkarray(); i=rand(1:D.ngenes); j=rand(1:D.nli); k=rand(1:D.nti))

# Align a vector of "<number> <unit>" strings on the decimal point. Units may
# differ (adaptive ns/us/ms, KiB/MiB) -- only the dots line up vertically; the
# unit trails. Splits each into integer part / fractional part / unit, then pads
# so the fractional parts (and thus the dots) start at the same column.
function decimal_align(strs)
    trip = map(strs) do s
        parts = split(s)
        num = parts[1]; unit = length(parts) > 1 ? parts[2] : ""
        d = findfirst('.', num)
        d === nothing ? (num, "", unit) : (num[1:d-1], num[d:end], unit)
    end
    wI = maximum(length(t[1]) for t in trip)
    wF = maximum(length(t[2]) for t in trip)
    # fixed-width "int.frac" prefix (dots line up), then the trailing unit;
    # left-align these columns so the prefix -- and thus the dots -- stay aligned
    [string(lpad(i, wI), rpad(f, wF), " ", u) for (i, f, u) in trip]
end

# resident size of the built index structures (what differs between index
# designs), via Base.summarysize. Exercises every access pattern first so all
# indices exist, then reports the core and its parts. Written to be agnostic to
# the core's fields so it works across index designs.
function index_footprint()
    core = parent(A).core
    A[Gene=At(D.genes[1])]; A[Lineage=At(D.lineages[1])]; A[Time=At(D.times[1])]
    A[Lineage=At(D.lineages[1]), Gene=At(D.genes[1])]
    A[Time=At(D.times[1]), Gene=At(D.genes[1])]
    A[Time=At(D.times[1]), Lineage=At(D.lineages[1])]
    A[1, 1, 1]
    mib(x) = Base.summarysize(x) / 2^20
    # the core's parts, then the shared-core subtotal, the per-layer values, and
    # the grand TOTAL (one layer's backing array = core + values)
    items = Tuple{String,Float64}[]
    hasproperty(core, :poscols) && push!(items, ("sorted poscols", mib(core.poscols)))
    if hasproperty(core, :scalarindex) && core.scalarindex[] !== nothing
        push!(items, ("scalar index", mib(core.scalarindex[])))
    end
    for (subset, idx) in sort(collect(core.indices); by = first)
        push!(items, ("index $(subset) ($(length(idx)) keys)", mib(idx)))
    end
    push!(items, ("core (shared subtotal)", mib(core)))
    push!(items, ("values (per layer)", mib(parent(A).values)))
    push!(items, ("TOTAL (core + values)", mib(parent(A))))
    sizes = decimal_align([@sprintf("%.2f MiB", last(it)) for it in items])
    pretty_table(hcat(first.(items), sizes);
                 column_labels = ["index footprint", "size"], alignment = [:l, :l])
end

# When run as a script (not via PkgBenchmark), execute and print time, per-call
# allocations, and the index footprint.
if abspath(PROGRAM_FILE) == @__FILE__
    println("rows: ", nrow(D.tab), "  (", D.ngenes, " genes x ", D.nli, " lineages x ",
            D.nti, " times, ", round(100*nrow(D.tab)/(D.ngenes*D.nli*D.nti)), "% filled)\n")
    results = run(SUITE; verbose=true)

    lv      = sort(BenchmarkTools.leaves(results); by = pt -> join(pt[1], "/"))
    names   = [join(p, "/") for (p, _) in lv]
    times   = decimal_align([BenchmarkTools.prettytime(median(t).time)     for (_, t) in lv])
    allocsz = decimal_align([BenchmarkTools.prettymemory(median(t).memory) for (_, t) in lv])
    nallocs = [median(t).allocs for (_, t) in lv]
    println()
    pretty_table(hcat(names, times, allocsz, nallocs);
                 column_labels = ["benchmark", "time", "alloc", "allocs"],
                 alignment = [:l, :l, :l, :r])
    println()
    index_footprint()
end
