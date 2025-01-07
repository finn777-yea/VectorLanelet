using Test
using LinearAlgebra
using VectorLanelet

@testset "SpatialAttention" begin
    @testset "Simple 2D case" begin
        agt_ctrs = [0.0 1.0;   # x coordinates
                    0.0 1.0]    # y coordinates
        ctx_ctrs = [0.0 2.0;   # x coordinates
                    1.0 2.0]    # y coordinates
        threshold = 1.5
        
        distances, mask = create_distance_mask(agt_ctrs, ctx_ctrs, threshold)
        expected_mask = [true true;    # Context 1 to Agents (1,2)
                        false true]    # Context 2 to Agents (1,2)
        
        @test mask == expected_mask
        @test distances[1,1] ≈ 1.0  # Distance from context 1 to agent 1
        @test distances[2,2] ≈ 1.4142135623730951  # Distance from context 2 to agent 2 (√2)
    end

    @testset "Larger grid case" begin
        agt_ctrs = [0.0  1.0  2.0  3.0;    # x coordinates of 4 agents
                    0.0  1.0  2.0  3.0]     # y coordinates of 4 agents
        
        ctx_ctrs = [0.0  1.0  2.0  3.0  4.0  0.5  1.5  2.5  3.5  4.5;    # x coordinates of 10 contexts
                    0.0  1.0  2.0  3.0  4.0  0.5  1.5  2.5  3.5  4.5]     # y coordinates of 10 contexts
        
        threshold = 1.0
        
        distances, mask = create_distance_mask(agt_ctrs, ctx_ctrs, threshold)
        
        expected_mask = [
            true  false false false;  # Context 1 (0,0)
            false true  false false;  # Context 2 (1,1)
            false false true  false;  # Context 3 (2,2)
            false false false true;   # Context 4 (3,3)
            false false false false;  # Context 5 (4,4)
            true  true  false false;  # Context 6 (0.5,0.5)
            false true  true  false;  # Context 7 (1.5,1.5)
            false false true  true;   # Context 8 (2.5,2.5)
            false false false true;   # Context 9 (3.5,3.5)
            false false false false   # Context 10 (4.5,4.5)
        ]
        
        @test mask == expected_mask
    end

    @testset "Input validation" begin
        # Test invalid dimensions
        invalid_agt_ctrs = [0.0 1.0 2.0;    # 3 rows instead of 2
                           0.0 1.0 2.0;
                           0.0 1.0 2.0]
        valid_ctx_ctrs = [0.0 1.0;
                         0.0 1.0]
        
        @test_throws ArgumentError create_distance_mask(invalid_agt_ctrs, valid_ctx_ctrs, 1.0)
        
        # Test negative threshold
        valid_agt_ctrs = [0.0 1.0;
                         0.0 1.0]
        
        @test_throws ArgumentError create_distance_mask(valid_agt_ctrs, valid_ctx_ctrs, -1.0)
    end
end
