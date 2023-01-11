@testset "Leray constructor             " begin
    # initialise variables and arrays
    Ny = rand(3:50)
    Nz = rand(3:50)
    Nt = rand(3:50)
    y = rand(Float64, Ny)
    Dy = rand(Float64, (Ny, Ny))
    Dy2 = rand(Float64, (Ny, Ny))
    ws = rand(Float64, Ny)
    ω = abs(randn())
    β = abs(randn())

    # initialise grid and field
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    U = spectralfield(grid)
    u = physicalfield(grid)
    U_vec = vectorfield(grid)
    u_vec = vectorfield(grid; field_type=:physical)

    # construct projection type
    @test Leray!(U) isa Leray!{typeof(U)}
    @test Leray!(U_vec) isa Leray!{typeof(U)}
    @test Leray!(grid) isa Leray!{typeof(U)}

    # catch errors
    @test_throws MethodError Leray!(U, rand(Float64, (Ny, Nz)))
end

@testset "Leray incompressible field    " begin
    # construct incompressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = chebws(Dy)
    ω = 1.0
    β = 1.0
    u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u = vectorfield(grid, u_fun, v_fun, w_fun)
    U = vectorfield(grid)
    FFT! = FFTPlan!(grid; flags=ESTIMATE)
    FFT!(U, u)
    U_aux = copy(U)

    # check divergence is zero
    dVdy = spectralfield(grid)
    dWdz = spectralfield(grid)
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div = dVdy + dWdz
    @test norm(div) < 1e-12

    # initialise projector
    leray! = Leray!(U)

    # perform projection
    leray!(U)

    # check vector field didn't change
    @test U ≈ U_aux
end

@testset "Leray compressible field      " begin
    # construct compressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = chebws(Dy)
    ω = 1.0
    β = 1.0
    u_fun(y, z, t) = (1 - y^2)*exp(cos(z))*atan(sin(t))
    v_fun(y, z, t) = sin(π*y)*exp(sin(z))*atan(cos(t))
    w_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*exp(cos(t))
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u = vectorfield(grid, u_fun, v_fun, w_fun)
    U = vectorfield(grid)
    FFT! = FFTPlan!(grid; flags=ESTIMATE)
    FFT!(U, u)

    # check divergence is non-zero
    dVdy = spectralfield(grid)
    dWdz = spectralfield(grid)
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div = dVdy + dWdz
    @test norm(div) > 1e-3

    # initialise projector
    leray! = Leray!(U)

    # perform projection
    leray!(U)

    # check divergence of new vector field
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div .= dVdy .+ dWdz
    @test norm(div) < 1e-6
end
