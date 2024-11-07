export NesterovSolver

mutable struct NesterovSolver{T, V} <: SolverCore.AbstractOptimizationSolver 
    # T is type for numbers and V is container type, for example, Vector{T}...
    x::V
    g::V
    alpha::T
end


function NesterovSolver(nlp::AbstractNLPModel{T, V}; alpha = 1e-3) where {T, V}
    x = copy(nlp.meta.x0) # iterate
    g = similar(nlp.meta.x0) # gradient

    return NesterovSolver{T, V}(x, g, T(alpha))
end

function SolverCore.solve!(solver::NesterovSolver, 
    nlp::AbstractNLPModel{T, V}, 
    stats::GenericExecutionStats{T,V} ; 
        callback = (args...) -> nothing,
        x::V = nlp.meta.x0,
        atol::T = √eps(T),
        rtol::T = √eps(T),
        alpha_max::T = 1 / eps(T),
        max_time::Float64 = 30.0,
        max_eval::Int = -1,
        max_iter::Int = typemax(Int),
        verbose::Int = 0) where {T, V}
    # Solver Core function                   
    unconstrained(nlp) || error("Gradient Descent should only be called on unconstrained problems.")

    reset!(stats)
    start_time = time()
    set_time!(stats, 0.0)

    x_k = solver.x .= x
    ∇fk = solver.g

    # ietration counter
    set_iter!(stats, 0)

    # objective evaluation
    f0 = obj(nlp, x_k)
    set_objective!(stats, f0)

    grad!(nlp, x_k, ∇fk)
    norm_∇fk = norm(∇fk)
    set_dual_residual!(stats, norm_∇fk)

    # Stopping criterion: 
    fmin = min(-one(T), f0) / eps(T)
    unbounded = f0 < fmin

    ϵ = atol + rtol * norm_∇fk
    optimal = norm_∇fk ≤ ϵ

    step_param_name = "α"

    if optimal
        @info("Optimal point found at initial point")
        @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" step_param_name
        @info @sprintf "%5d  %9.2e  %7.1e  %7.1e" stats.iter stats.objective norm_∇fk solver.alpha
        else
        if verbose > 0 && mod(stats.iter, verbose) == 0
            step_param = solver.alpha
            @info @sprintf "%5s  %9s  %7s  %7s " "iter" "f" "‖∇f‖" step_param_name 
            infoline =
            @sprintf "%5d  %9.2e  %7.1e  %7.1e " stats.iter stats.objective norm_∇fk step_param
        end
    end

    set_status!(
    stats,
    get_status(
    nlp,
    elapsed_time = stats.elapsed_time,
    optimal = optimal,
    unbounded = unbounded,
    max_eval = max_eval,
    iter = stats.iter,
    max_iter = max_iter,
    max_time = max_time,
    ),
    )

    callback(nlp, solver, stats)

    done = stats.status != :unknown

    while !done

        x_k .-= solver.alpha .* ∇fk
        fk = obj(nlp, x_k)
        unbounded = fk < fmin

        set_objective!(stats, fk)

        grad!(nlp, x_k, ∇fk)
        norm_∇fk = norm(∇fk)

        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)
        set_dual_residual!(stats, norm_∇fk)
        optimal = norm_∇fk ≤ ϵ

        if verbose > 0 && mod(stats.iter, verbose) == 0
            @info infoline
            step_param = solver.alpha
            infoline =
            @sprintf "%5d  %9.2e  %7.1e  %7.1e " stats.iter stats.objective norm_∇fk step_param 
        end

        set_status!(
        stats,
        get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        unbounded = unbounded,
        max_eval = max_eval,
        iter = stats.iter,
        max_iter = max_iter,
        max_time = max_time,
        ),
        )

        callback(nlp, solver, stats)

        done = stats.status != :unknown
    end

    set_solution!(stats, x_k)
    return stats

end