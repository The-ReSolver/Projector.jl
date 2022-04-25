# This file contains the functionality of the Leray projection of a
# compressible vector field in a rotating-plane couette flow.

struct Leray!{S}
    cache::Vector{S}
    lapl::Laplace

    function Leray!(U::AbstractArray{Complex{T}, 3}) where {T}
        # initialised cached arrays
        cache = [similar(U) for i in 1:6]

        # extract grid
        grid = get_grid(U)

        # initialise laplacian
        lapl = Laplace(size(grid)[1], size(grid)[2], get_β(grid), get_Dy2(grid), get_Dy(grid))

        new{typeof(U)}(cache, lapl)
    end
end

Leray!(U::AbstractVector{S}) where {S} = Leray!(U[1])

function (f::Leray!{S})(U::AbstractVector{S}) where {S}
    # assign aliases
    ϕ = f.cache[1]
    dϕdy = f.cache[2]
    dϕdz = f.cache[3]
    dVdy = f.cache[4]
    dWdz = f.cache[5]
    rhs = f.cache[6]

    # compute divergence of given vector field
    ddy!(U[2], dVdy)
    ddz!(U[3], dWdz)
    rhs .= .-dVdy .- dWdz

    # solve poisson equation
    solve!(ϕ, f.lapl, rhs)

    # compute gradient of scalar field
    ddy!(ϕ, dϕdy)
    ddz!(ϕ, dϕdz)

    # project original field
    U[2] .+= dϕdy
    U[3] .+= dϕdz

    return U
end
