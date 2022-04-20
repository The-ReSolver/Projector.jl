module Projector

using Fields
using PoissonSolver

export Leray!
export SlipCorrector!

include("leray.jl")
include("noslip.jl")

end
