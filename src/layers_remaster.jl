function create_residual_block(n_in, n_out; kernel_size=3, stride, norm="GN", ng=32, act=true)
    filter = (kernel_size,)
    
    # Main convolution branch
    main_branch = Chain(
        Conv(filter, n_in => n_out, stride=stride, pad=SamePad()),
        norm == "GN" ? GroupNorm(n_out, gcd(ng, n_out)) : BatchNorm(n_out),
        relu,
        Conv(filter, n_out => n_out, stride=1, pad=SamePad()),
        norm == "GN" ? GroupNorm(n_out, gcd(ng, n_out)) : BatchNorm(n_out)
    )
    
    # Identity/downsample branch
    if stride != 1 || n_out != n_in
        identity_branch = Chain(
            Conv((1,), n_in=>n_out, stride=stride),
            norm == "GN" ? GroupNorm(n_out, gcd(ng, n_out)) : BatchNorm(n_out)
        )
    else
        identity_branch = identity
    end
    
    # Combine branches with skip connection
    residual = Chain(
        Parallel(+, main_branch, identity_branch),
        act ? relu : identity
    )
    
    return residual
end