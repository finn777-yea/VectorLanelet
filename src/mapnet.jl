struct MapNet
    cfg
    input
    seg
    fuse
end

Flux.@layer MapNet

function MapNet(config)
    n_map = config["n_map"]
    n_scales = config["num_scales"]
    norm = "GN"
    ng = 1

    input = Chain(
        Dense(64=>n_map, relu),
        Linear(n_map, n_map, norm=norm, ng=ng, act=false)
    )

    seg = Chain(
        Dense(2=>n_map, relu),
        Linear(n_map, n_map, norm=norm, ng=ng, act=false)
    )

    fuse = Dict()
    ks = ["ctr", "norm", "ctr2", "heteroconv"]
    for key in ks
        fuse[key] = []
    end

    # Aggregate heterogenously
    left_rel = (:lanelet, :left, :lanelet)
    right_rel = (:lanelet, :right, :lanelet)
    adj_left_rel = (:lanelet, :adj_left, :lanelet)
    adj_right_rel = (:lanelet, :adj_right, :lanelet)
    suc_rel = (:lanelet, :suc, :lanelet)

    heteroconv = HeteroGraphConv(
            left_rel => GATConv(n_map=>n_map),
            right_rel => GATConv(n_map=>n_map),
            adj_left_rel => GATConv(n_map=>n_map),
            adj_right_rel => GATConv(n_map=>n_map),
            suc_rel => GATConv(n_map=>n_map)
        )

    for i in 1:4
        for key in keys(fuse)
            if key in ["norm"]
                push!(fuse[key], GroupNorm(n_map, gcd(ng, n_map)))
            elseif key in ["ctr2"]
                push!(fuse[key], Linear(n_map, n_map, norm=norm, ng=ng))
            elseif key in ["heteroconv"]
                push!(fuse[key], heteroconv)
            else
                push!(fuse[key], Dense(n_map=>n_map))
            end
        end
    end

    for key in keys(fuse)
        fuse[key] = Chain(fuse[key]...)
    end

    MapNet(config, input, seg, fuse)
end

"""
Expect heterograph with 5 relation types
Original Input:
    - g[:lanelet].ctrs: center position of the lane
    - g[:lanelet].feat: start/end position of the lane 

New Input:
    - g[:lanelet].x:(64, num_lanelets)
"""

function (mapnet::MapNet)(g::GNNHeteroGraph)
    # TODO: check if node features are empty

    ### Node features
    feat = mapnet.input(g[:lanelet].x)
    feat = relu.(feat)

    ### Fusion aggregation
    # TODO: check if a certain relation exists

    # feat: (128, num_nodes)
    res = feat

    for i in eachindex(mapnet.fuse["ctr"])
        temp = (;lanelet = mapnet.fuse["ctr"][i](feat))
        temp = mapnet.fuse["heteroconv"][i](g, temp).lanelet

        feat = mapnet.fuse["norm"][i](temp)
        feat = relu.(feat)
        feat = mapnet.fuse["ctr2"][i](feat)
        feat = feat + res
        feat = relu.(feat)

        res = feat
    end

    return feat
end