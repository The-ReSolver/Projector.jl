@testset "Slip corrector constructor    " begin
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
    U = SpectralField(grid)

    # construct correction type
    @test typeof(SlipCorrector!(U)) == SlipCorrector!{typeof(U)}
end

@testset "Slip corrector without slip   " begin
    # construct incompressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = quadweights(y, 2)
    ω = 1.0
    β = 1.0
    u_fun(y, z, t) = sin(π*y)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=ESTIMATE)
    FFT!(U, u)
    U_aux = copy(U)

    # initialise slip corrector
    slipcorrector! = SlipCorrector!(U)

    # perform correction
    slipcorrector!(U)

    # check field didn't change
    @test U ≈ U_aux
end

@testset "Slip corrector with slip      " begin
    # construct incompressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = quadweights(y, 2)
    ω = 1.0
    β = 1.0
    u_fun(y, z, t) = (sin(π*y) + 1)*exp(cos(z))*sin(t)
    v_fun(y, z, t) = (cos(π*y) + 1)*cos(z)*sin(t)
    w_fun(y, z, t) = π*sin(π*y)*sin(z)*sin(t)
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=ESTIMATE)
    FFT!(U, u)

    # initialise slip corrector
    slipcorrector! = SlipCorrector!(U)

    # perform correction
    slipcorrector!(U)

    # check vector field is still incompressible
    dVdy = SpectralField(grid)
    dWdz = SpectralField(grid)
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div = dVdy + dWdz
    @test norm(div) < 1e-12

    # check no-slip boundary condition
    @test U[1][1, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[1][end, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[2][1, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[2][end, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[3][1, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[3][end, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
end

@testset "Slip corrector with leray     " begin
    # construct incompressible vector field
    Ny = 64; Nz = 64; Nt = 64
    y = chebpts(Ny)
    Dy = chebdiff(Ny)
    Dy2 = chebddiff(Ny)
    ws = quadweights(y, 2)
    ω = 1.0
    β = 1.0
    u_fun(y, z, t) = (y^2)*exp(cos(z))*atan(sin(t))
    v_fun(y, z, t) = sin(π*y)*exp(sin(z))*atan(cos(t))
    w_fun(y, z, t) = cos(π*y)*cos(z)*exp(cos(t))
    grid = Grid(y, Nz, Nt, Dy, Dy2, ws, ω, β)
    u = VectorField(PhysicalField(grid, u_fun),
                    PhysicalField(grid, v_fun),
                    PhysicalField(grid, w_fun))
    U = VectorField(grid)
    FFT! = FFTPlan!(grid; flags=ESTIMATE)
    FFT!(U, u)

    # initialise leray projection and slip corrector
    leray! = Leray!(U, u)
    slipcorrector! = SlipCorrector!(U)

    # perform projection and correction
    leray!(U)
    slipcorrector!(U)

    # check vector field is still incompressible
    dVdy = SpectralField(grid)
    dWdz = SpectralField(grid)
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    div = dVdy + dWdz
    @test norm(div) < 1e-6

    # check no-slip boundary condition
    @test U[1][1, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[1][end, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[2][1, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    @test U[2][end, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-12
    # NOTE: the allowable tolerance for the tests below depend on τ and y resolution
    @test U[3][1, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-7
    @test U[3][end, :, :] ≈ zeros((Nz >> 1) + 1, Nt) atol=1e-7
end
