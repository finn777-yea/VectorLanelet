using Flux
using BenchmarkTools
using GraphNeuralNetworks

a = rand(2, 10) |> gpu
maska = a .<= 0.5
@btime findall(x -> x.<=0.5, $a)        # 89.138 us

b = copy(a) |> cpu
maskb = b .<= 0.5
@btime findall(x -> x.<=0.5, $b)        # 175.824 ns

ci = CartesianIndex(1, 1)
cis = [CartesianIndex(i,2i) for i in 1:5]
cis = Tuple.(cis)
map(collect, cis)
reduce(hcat, cis)
# Convert vector of tuples to 2D array
tuples_to_matrix = reduce(hcat, collect.(cis))
@show tuples_to_matrix # Shows a 2×5 Matrix{Int64}

# Alternative method using comprehension
matrix_comp = [x[i] for x in cis, i in 1:2]
@show matrix_comp # Shows a 5×2 Matrix{Int64} (transposed version)

function indices_to_matrix(indices::Vector{CartesianIndex{2}})
    tuple_idc = map(Tuple, indices)
    return reduce(hcat, map(collect, tuple_idc))
end

indices_to_matrix(cis)

src = [1, 2, 3, 4]
dst = [5, 6, 7, 8]

@btime g = GNNGraph($src, $dst)

src_cuda = src |> gpu
dst_cuda = dst |> gpu

@btime g_cuda = GNNGraph($src_cuda, $dst_cuda)

