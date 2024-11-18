"""
Use first two points on a lanelet centerline as feature, last point on the same lanelet as label
    - agt_features : (timesteps, channels)
    - timesteps = 2; channels = 2
"""

using VectorLanelet
using Lanelet2
using Lanelet2.Routing
using Lanelet2.Core
using AutomotiveSimulator

# Load the lanelet map
example_file = joinpath(@__DIR__, "../res","location0.osm")
projector = Projection.UtmProjector(Lanelet2.Io.Origin(49, 8.4))
llmap = Lanelet2.Io.load(example_file, projector)

traffic_rules = TrafficRules.create(TrafficRules.Locations.Germany, TrafficRules.Participants.Vehicle)
rg = RoutingGraph(llmap, traffic_rules)
llmap = rg.passableLaneletSubmap()
lanelet_roadway = LaneletRoadway(llmap)

# Iterate over lanelets to retrieve the corresponding positions
agt_features = []
pos_agt = []
labels = []
for lanelet in values(lanelet_roadway.lanelets)
    curve = lanelet.curve
    point1 = [curve[1].pos.x, curve[1].pos.y]
    point2 = [curve[2].pos.x, curve[2].pos.y]       # the current position of the agt
    point_e = [curve[end].pos.x, curve[end].pos.y]

    push!(agt_features, hcat(point1, point2))
    push!(pos_agt, point2)
    push!(labels, point_e)
end

# Split data into training and test sets
n_train = 70
agt_features_train = agt_features[1:n_train]
agt_features_test = agt_features[n_train+1:end]
agt_pos_train = pos_agt[1:n_train]
agt_pos_test = pos_agt[n_train+1:end]
labels_train = labels[1:n_train]
labels_test = labels[n_train+1:end]
# TODO: apply DataLoader

# Concatenate the features as before, but separately for train and test
agt_features_train = cat(agt_features_train..., dims=3)
agt_features_test = cat(agt_features_test..., dims=3)


### Simplified actor encoder
config = Dict{String, Any}()
config["n_actor"] = 128     # Actor embedding output dimension
config["din_actor"] = 2

actornet = ActorNet_Simp(config)
emb_actor = actornet(agt_features_train)
@assert size(emb_actor) == (128, 70)