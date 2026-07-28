# Benchmark suite for SparseDimArrays.
#
# Run standalone:
#     julia --project=benchmark benchmark/benchmarks.jl
# or with PkgBenchmark (uses the `SUITE` defined here):
#     using PkgBenchmark; benchmarkpkg("SparseDimArrays")
#
# The suite covers construction and every access pattern the design cares about:
# the scalar (full-key) lookup, single-dimension slices, a pair slice, a full
# materialize, and the *cold* cost of building an index on first use (the part
# the sorted-prefix work changes).

using BenchmarkTools, SparseDimArrays, DimensionalData, DataFrames, Random

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
mkarray() = sparsedimarray(D.tab, (:iGE, :iLI, :TI), :val, dims, NaN32;
                           precoded=(true, true, false))

const SUITE = BenchmarkGroup()

# --- construction ---
SUITE["construct"] = BenchmarkGroup()
SUITE["construct"]["array"] = @benchmarkable mkarray()
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

# When run as a script (not via PkgBenchmark), execute and print results.
if abspath(PROGRAM_FILE) == @__FILE__
    println("rows: ", nrow(D.tab), "  (", D.ngenes, " genes x ", D.nli, " lineages x ",
            D.nti, " times, ", round(100*nrow(D.tab)/(D.ngenes*D.nli*D.nti)), "% filled)\n")
    results = run(SUITE; verbose=true)
    println()
    show(stdout, MIME"text/plain"(), results)
    println()
end
