using Flux
using BenchmarkTools
a = rand(2, 10) |> gpu
maska = a .<= 0.5
@btime findall(x -> x.<=0.5, $a)        # 89.138 us

b = copy(a) |> cpu
maskb = b .<= 0.5
@btime findall(x -> x.<=0.5, $b)        # 175.824 ns