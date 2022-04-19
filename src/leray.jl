# This file contains the functionality of the Leray projection of a
# compressible vector field in a rotating-plane couette flow.

export Leray!

struct Leray!{S}
    cache::Vector{S}
    lapl::Laplace

    function Leray!(U::S, u::P) where {T, S<:AbstractArray{Complex{T}, 3}, P<:AbstractArray{T, 3}}
        # check fields have compatible grids
        U.grid == u.grid || throw(ArgumentError("Fields are on different grids!"))

        # initialised cached arrays
        cache = [similar(U) for i in 1:6]

        # initialise laplacian
        lapl = Laplace(size(u)[1], size(u)[2], U.grid.dom[2], U.grid.Dy[2], U.grid.Dy[1])

        new{S}(cache, lapl)
    end
end

Leray!(U::VS, u::VP) where {S, P, VS<:AbstractVector{S}, VP<:AbstractVector{P}} = Leray!(U[1], u[1])
Leray!(grid::G) where {G} = Leray!(SpectralField(grid), PhysicalField(grid))

function (f::Leray!{S})(U::V) where {T, S, V<:AbstractVector{S}}
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
