using Test
using SparseDimArrays
using DimensionalData
using DataFrames
using Arrow

const Gene = Dim{:Gene}
const Lineage = Dim{:Lineage}
const Time = Dim{:Time}

function oracle(rows, valuecol, missingval, dimvals, dimcols, keys)
    out = fill(missingval, map(length, dimvals))
    for idx in CartesianIndices(out)
        wanted = ntuple(d -> dimvals[d][idx[d]], length(dimvals))
        r = findfirst(rows) do row
            all(ntuple(d -> getproperty(row, dimcols[d]) == wanted[d], length(dimvals)))
        end
        isnothing(r) || (out[idx] = getproperty(rows[r], valuecol))
    end
    out
end

@testset "SparseDimArrays" begin

@testset "3-d (gene, lineage, time), unsorted dims, sparse" begin
    genes    = ["g11", "g10", "g12"]        # deliberately NOT sorted
    lineages = ["lB", "lA", "lC", "lD"]
    times    = Int16[100, 200, 300, 400, 500]

    rows = NamedTuple[]
    for (ig, g) in enumerate(genes), (il, l) in enumerate(lineages), (it, t) in enumerate(times)
        (ig + il + it) % 3 == 0 && continue   # ~sparse
        push!(rows, (gene=g, lineage=l, time=t, val=Float32(ig*100+il*10+it), samp=string(g,"-",l,"-",t)))
    end
    tab = DataFrame(rows)

    dims = (Gene(genes), Lineage(lineages), Time(times))
    A  = sparsedimarray(tab, (:gene, :lineage, :time), :val, dims, NaN32)
    As = sparsedimarray(tab, (:gene, :lineage, :time), :samp, dims, "")

    ora = (valuecol, missingval) -> oracle(rows, valuecol, missingval, (genes, lineages, times),
                                            (:gene, :lineage, :time), nothing)

    @test isequal(collect(A[:, :, :]), ora(:val, NaN32))
    @test isequal(collect(A[Gene=At("g10")]), ora(:val, NaN32)[2, :, :])
    @test isequal(collect(A[Lineage=At("lA")]), ora(:val, NaN32)[:, 2, :])
    @test isequal(collect(A[Time=At(Int16(300))]), ora(:val, NaN32)[:, :, 3])
    @test isequal(collect(A[Lineage=At("lA"), Gene=At("g10")]), ora(:val, NaN32)[2, 2, :])
    @test isequal(collect(As[Gene=At("g10"), Time=At(Int16(300))]), ora(:samp, "")[2, :, 3])
    @test isequal(collect(As[Lineage=At("lA"), Time=At(Int16(300))]), ora(:samp, "")[:, 2, 3])
    @test isequal(A[2, 2, 3], ora(:val, NaN32)[2, 2, 3])
    @test isequal(collect(A[Gene=At(["g11", "g12"]), Lineage=At(["lB", "lC"])]),
                  ora(:val, NaN32)[[1, 3], [1, 3], :])
    @test isequal(collect(A[Gene=At(["g11", "g12"]), Lineage=At(["lB", "lC"]), Time=At(Int16[100, 300])]),
                  ora(:val, NaN32)[[1, 3], [1, 3], [1, 3]])

    r = A[Gene=At("g10")]
    @test r isa DimArray
    @test size(r) == (4, 5)
    @test hasproperty(r, :data)
    @test isequal(collect(r[Time=At(Int16(200))]), ora(:val, NaN32)[2, :, 2])
    @test typeof(set(r, Lineage=>DimensionalData.Unordered)) <: DimArray
    # cat over a DISJOINT lineage set, as a caller might merge extra lineages
    # into the result (cf. concatenating with lineages absent from this table)
    extra = fill(NaN32, Lineage(["lE", "lF"]), Time(times))
    combined = cat(set(r, Lineage=>DimensionalData.Unordered), extra, dims=Lineage)
    @test size(combined) == (6, 5)

    # index cache: lazily built on demand, and reused (not rebuilt) on repeat access
    B = sparsedimarray(tab, (:gene, :lineage, :time), :val, dims, NaN32)
    Bcore = parent(B).core
    @test isempty(Bcore.indices)
    B[Gene=At("g10")]
    @test haskey(Bcore.indices, (1,))
    cached = Bcore.indices[(1,)]
    B[Gene=At("g10")]
    @test Bcore.indices[(1,)] === cached   # same object: not rebuilt

    # eager indices built at construction match lazy ones
    C = sparsedimarray(tab, (:gene, :lineage, :time), :val, dims, NaN32; indices=((1,), (1,2)))
    Ccore = parent(C).core
    @test haskey(Ccore.indices, (1,))
    @test haskey(Ccore.indices, (1, 2))
    @test isequal(collect(C[Gene=At("g10")]), collect(A[Gene=At("g10")]))
    @test isequal(collect(C[Lineage=At("lA"), Gene=At("g10")]), collect(A[Lineage=At("lA"), Gene=At("g10")]))

    # precoded columns: gene/lineage stored as compact 1-based integer codes
    # (matching dimension position directly) instead of repeating name strings
    genemap = Dict(g => i for (i, g) in enumerate(genes))
    limap   = Dict(l => i for (i, l) in enumerate(lineages))
    codedrows = [(iGE=genemap[r.gene], iLI=limap[r.lineage], time=r.time, val=r.val) for r in rows]
    codedtab = DataFrame(codedrows)
    D = sparsedimarray(codedtab, (:iGE, :iLI, :time), :val, dims, NaN32; precoded=(true, true, false))
    @test isequal(collect(D[:, :, :]), ora(:val, NaN32))
    @test isequal(collect(D[Gene=At("g10")]), collect(A[Gene=At("g10")]))
    @test isequal(collect(D[Lineage=At("lA"), Gene=At("g10")]), collect(A[Lineage=At("lA"), Gene=At("g10")]))
    @test isequal(D[2, 2, 3], A[2, 2, 3])
end

@testset "sparsedimstack: shared core across value columns" begin
    genes    = ["g11", "g10", "g12"]
    lineages = ["lB", "lA", "lC", "lD"]
    times    = Int16[100, 200, 300, 400, 500]
    rows = NamedTuple[]
    for (ig, g) in enumerate(genes), (il, l) in enumerate(lineages), (it, t) in enumerate(times)
        (ig + il + it) % 3 == 0 && continue
        push!(rows, (gene=g, lineage=l, time=t, val=Float32(ig*100+il*10+it), samp=string(g,"-",l,"-",t)))
    end
    tab = DataFrame(rows)
    dims = (Gene(genes), Lineage(lineages), Time(times))
    ora = (valuecol, missingval) -> oracle(rows, valuecol, missingval, (genes, lineages, times),
                                            (:gene, :lineage, :time), nothing)

    stack = sparsedimstack(tab, (:gene, :lineage, :time), (:val, :samp), dims, (NaN32, ""))
    @test stack isa DimStack
    @test stack.val isa DimArray
    @test stack.samp isa DimArray

    # one shared core underlying both layers -- not two independently-built ones
    @test parent(stack.val).core === parent(stack.samp).core

    @test isequal(collect(stack.val[Gene=At("g10")]), ora(:val, NaN32)[2, :, :])
    @test isequal(collect(stack.samp[Lineage=At("lA"), Time=At(Int16(300))]), ora(:samp, "")[:, 2, 3])

    # indexing the *stack* itself slices every layer together
    sliced = stack[Lineage=At("lA"), Gene=At("g10")]
    @test sliced isa DimStack
    @test isequal(collect(sliced.val), ora(:val, NaN32)[2, 2, :])
    @test isequal(collect(sliced.samp), ora(:samp, "")[2, 2, :])

    # a fully-scalar selector on the stack collapses to a plain NamedTuple
    scalarsel = stack[Lineage=At("lA"), Gene=At("g10"), Time=At(Int16(300))]
    @test scalarsel isa NamedTuple
    @test scalarsel.val == ora(:val, NaN32)[2, 2, 3]
    @test scalarsel.samp == ora(:samp, "")[2, 2, 3]
end

@testset "minimal-width index types" begin
    genes    = ["g$i" for i in 1:300]     # > 255 -> needs UInt16, not UInt8
    lineages = ["l$i" for i in 1:5]        # fits UInt8
    times    = Int32.(1:70000)              # > 65535 -> needs UInt32
    rows = [(gene=genes[1], lineage=lineages[1], time=times[i], val=1.0f0) for i in 1:70000]
    tab = DataFrame(rows)
    dims = (Gene(genes), Lineage(lineages), Time(times))
    A = sparsedimarray(tab, (:gene, :lineage, :time), :val, dims, NaN32)
    core = parent(A).core

    @test core.postypes == (UInt16, UInt8, UInt32)
    @test core.rowtype == UInt32   # 70000 rows -> exceeds UInt16, needs UInt32

    A[Gene=At(genes[1])]        # trigger index (1,)
    @test keytype(core.indices[(1,)]) == Tuple{UInt16}
    @test valtype(core.indices[(1,)]) == Vector{UInt32}

    A[Lineage=At(lineages[1]), Gene=At(genes[1])]   # trigger index (1,2)
    @test keytype(core.indices[(1, 2)]) == Tuple{UInt16,UInt8}

    # values are still correct across the narrow-type boundary, not just narrow
    @test A[1, 1, 70000] == 1.0f0
end

@testset "2-d (generality check)" begin
    Row = Dim{:Row}
    Col = Dim{:Col}
    rows = ["r1","r2","r3"]
    cols = ["c1","c2","c3","c4"]
    data = NamedTuple[]
    for (i,r) in enumerate(rows), (j,c) in enumerate(cols)
        (i+j) % 2 == 0 && continue
        push!(data, (row=r, col=c, val=Float64(i*10+j)))
    end
    tab = DataFrame(data)
    A = sparsedimarray(tab, (:row, :col), :val, (Row(rows), Col(cols)), NaN)
    ora = oracle(data, :val, NaN, (rows, cols), (:row, :col), nothing)
    @test isequal(collect(A[:, :]), ora)
    @test isequal(collect(A[Row=At("r2")]), ora[2, :])
    @test isequal(A[2,3], ora[2,3])
end

@testset "Arrow round-trip" begin
    genes = ["g11", "g10", "g12"]
    lineages = ["lB", "lA", "lC"]
    times = Int16[10, 20, 30]
    rows = [(gene=g, lineage=l, time=t, val=Float32(i))
            for (i,(g,l,t)) in enumerate(Iterators.product(genes, lineages, times))][:]
    tab = DataFrame(rows)
    mktempdir() do dir
        path = joinpath(dir, "t.arrow")
        Arrow.write(path, tab)
        loaded = Arrow.Table(path)
        A = sparsedimarray(loaded, (:gene, :lineage, :time), :val,
                            (Gene(genes), Lineage(lineages), Time(times)), NaN32)
        @test A[Gene=At("g10"), Lineage=At("lA"), Time=At(Int16(20))] isa Float32
        @test !isnan(A[Gene=At("g10"), Lineage=At("lA"), Time=At(Int16(20))])
    end
end

@testset "thread safety of lazy index construction" begin
    genes = ["g$i" for i in 1:20]
    lineages = ["l$i" for i in 1:20]
    times = Int16.(1:20)
    rows = [(gene=g, lineage=l, time=t, val=Float32(1))
            for (g,l,t) in Iterators.product(genes, lineages, times)][:]
    tab = DataFrame(rows)
    A = sparsedimarray(tab, (:gene, :lineage, :time), :val,
                        (Gene(genes), Lineage(lineages), Time(times)), NaN32)
    results = Vector{Any}(undef, 20)
    Threads.@threads for i in 1:20
        results[i] = collect(A[Gene=At(genes[mod1(i,20)])])
    end
    @test all(r -> all(==(1f0), r), results)
end

end
