export fomo, FomoSolver, FoSolver, fo, R2, TR, tr_step, r2_step

abstract type AbstractFirstOrderSolver <: AbstractOptimizationSolver end

abstract type AbstractFOMethod end

include("fomo.jl")

include("gradient_descent.jl")