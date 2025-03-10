using Flux
using Statistics

channels = 2
x = randn(Float32, channels, 3, 4)
bn = BatchNorm(channels)
Flux.trainmode!(bn)
y = bn(permutedims(x, (2,1,3)))
mean(y, dims=[1,3])
isapprox(mean(y, dims=[1,3]), zeros(1,2,1), atol=1e-6)       # normalize across 1,3 dims

ln = LayerNorm(channels)
y = ln(x)
mean(y, dims=1)
isapprox(mean(y, dims=1), zeros(1,3,4), atol=1e-6)
isapprox(mean(y, dims=1:2), zeros(1,1,4), atol=1e-6)

ln2 = LayerNorm(channels,3)
y = ln2(x)
!isapprox(mean(y, dims=1), zeros(1,3,4), atol=1e-6)
isapprox(mean(y, dims=1:2), zeros(1,1,4), atol=1e-6)

# G(number of groups)=1, GroupNorm = LayerNorm
# seperate the channels into 1 group
gn = GroupNorm(channels, 1)
y = gn(permutedims(x, (2,1,3)))
mean(y, dims=1:2)
isapprox(mean(y, dims=1:2), zeros(1,1,4), atol=1e-6)


# ------ LayerNorm issue -------
x = zeros32(3, 4)
ln = LayerNorm(3)
ln(x)