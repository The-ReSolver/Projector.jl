# This file contains the functionality required to impose the no-slip boundary
# conditions of a rotating-plane couette flow while keeping the flow
# incompressible.

# wall-normal functions with constant height
βₗ(y; τ=100) =  τ*(y + 1)*exp(1 - τ*(y + 1))
βᵤ(y; τ=100) = -τ*(y - 1)*exp(1 + τ*(y - 1))
dβₗdy(y; τ=100) = τ*exp(1 - τ*(y + 1)) - (τ^2)*(y + 1)*exp(1 - τ*(y + 1))
dβᵤdy(y; τ=100) = -τ*exp(1 + τ*(y - 1)) - (τ^2)*(y - 1)*exp(1 + τ*(y - 1))

struct SlipCorrector!{S}
    cache::Vector{S}

    function SlipCorrector!(U::AbstractArray{Complex{T}, 3}) where {T}
        # initialise cached arrays
        cache = [similar(U) for i in 1:4]

        new{typeof(U)}(cache)
    end
end

SlipCorrector!(U::AbstractVector{S}) where {S} = SlipCorrector!(U[1])
SlipCorrector!(grid::G) where {G} = SlipCorrector!(spectralfield(grid))

function (f::SlipCorrector!{S})(U::V) where {S, V<:AbstractVector{S}}
    # assign aliases
    ψ = f.cache[1]
    gx = f.cache[2]
    gy = f.cache[3]
    gz = f.cache[4]

    # compute streamfunction field
    for nt in 1:size(ψ)[3], nz in 1:size(ψ)[2], ny in 1:size(ψ)[1]
        ψ[ny, nz, nt] = -(U[3][end, nz, nt]/dβₗdy(-1))*βₗ(U[1].grid.y[ny]) - (U[3][1, nz, nt]/dβᵤdy(1))*βᵤ(U[1].grid.y[ny])
    end

    # compute correction fields from steamfunction
    ddy!(ψ, gz)
    ddz!(ψ, gy)

    # compute streamwise correction field
    for nt in 1:size(gx)[3], nz in 1:size(gx)[2], ny in 1:size(gx)[1]
        gx[ny, nz, nt] = -(U[1][end, nz, nt]/dβₗdy(-1))*dβₗdy(U[1].grid.y[ny]) - (U[1][1, nz, nt]/dβᵤdy(1))*dβᵤdy(U[1].grid.y[ny])
    end

    # modify given vector field
    U[1] .+= gx
    U[2] .-= gy
    U[3] .+= gz
end
