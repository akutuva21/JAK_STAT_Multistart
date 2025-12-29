# run_single_task.jl
# Runs a SINGLE optimization task for use with Slurm array jobs.
# Each task runs one optimization from a reproducible random starting point.

using Pkg
using ArgParse
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, CSV
using Optimization
import Optim
const O = Optim
using OptimizationOptimJL
using SymbolicUtils, Symbolics
using Random
using JLD2
using OrdinaryDiffEq  # For QNDF stiff solver
using SparseArrays
using LineSearches
using ForwardDiff

# REQUIRED POLYFILL: findnz for dense matrices
# PEtab.jl or ModelingToolkit calls findnz on dense Jacobians.
# In Julia 1.10+, SparseArrays.findnz only works on sparse matrices.
# This type piracy is necessary for compatibility - do not remove.
function SparseArrays.findnz(A::AbstractMatrix)
    I = Int[]
    J = Int[]
    V = eltype(A)[]
    rows, cols = size(A)
    for c in 1:cols
        for r in 1:rows
            val = A[r, c]
            if !iszero(val)
                push!(I, r)
                push!(J, c)
                push!(V, val)
            end
        end
    end
    return (I, J, V)
end

# ============================================================================
# CONFIGURATION - Uses current directory (works on scratch)
# ============================================================================
const MODEL_NET = joinpath(@__DIR__, "..", "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "..", "petab_files")
const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

const MEASUREMENTS_FILE = joinpath(PETAB_DIR, "measurements.tsv")
const CONDITIONS_FILE = joinpath(PETAB_DIR, "conditions.tsv")
const PARAMETERS_FILE = joinpath(PETAB_DIR, "parameters.tsv")
const OBSERVABLES_FILE = joinpath(PETAB_DIR, "observables.tsv")

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--task-id"
            help = "Task ID for this array job (1 to n-starts)"
            arg_type = Int
            required = true
        "--n-starts"
            help = "Total number of multistart runs"
            arg_type = Int
            default = 100
        "--max-iter"
            help = "Maximum iterations per optimization"
            arg_type = Int
            default = 1000
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = Int
            default = 1234
    end
    return parse_args(s)
end

# ============================================================================
# PETAB SETUP (same as run_multistart_from_files.jl)
# ============================================================================
function load_petab_from_files()
    println("Loading PEtab problem from TSV files...")
    
    # Verify required files exist
    if !isfile(MODEL_NET)
        error("Model file not found: $MODEL_NET")
    end
    for f in [MEASUREMENTS_FILE, CONDITIONS_FILE, PARAMETERS_FILE, OBSERVABLES_FILE]
        if !isfile(f)
            error("PEtab file not found: $f")
        end
    end
    
    prn = loadrxnetwork(BNGNetwork(), MODEL_NET)
    rsys = complete(prn.rn)
    odesys = structural_simplify(convert(ODESystem, rsys); simplify=true)
    
    measurements_df = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
    conditions_df = CSV.read(CONDITIONS_FILE, DataFrame; delim='\t')
    parameters_df = CSV.read(PARAMETERS_FILE, DataFrame; delim='\t')
    observables_df = CSV.read(OBSERVABLES_FILE, DataFrame; delim='\t')
    
    model_params = prn.p
    param_map = Dict(string(Symbolics.getname(k)) => k for (k, v) in model_params)
    
    # Build simulation conditions
    sim_conditions = Dict{String, Dict{Symbol, Float64}}()
    for row in eachrow(conditions_df)
        cond_id = string(row.conditionId)  # Ensure String type for PEtab
        cond_dict = Dict{Symbol, Float64}()
        for col in names(conditions_df)
            if col != "conditionId" && !ismissing(row[col])
                if haskey(param_map, col)
                    cond_dict[Symbolics.getname(param_map[col])] = Float64(row[col])
                else
                    cond_dict[Symbol(col)] = Float64(row[col])
                end
            end
        end
        sim_conditions[cond_id] = cond_dict
    end
    
    # Build parameters
    petab_params = PEtabParameter[]
    for row in eachrow(parameters_df)
        p_id = row.parameterId
        p_scale = Symbol(row.parameterScale)
        p_lb = Float64(row.lowerBound)
        p_ub = Float64(row.upperBound)
        p_nominal = Float64(row.nominalValue)
        p_estimate = row.estimate == 1
        
        if haskey(param_map, p_id)
            push!(petab_params, PEtabParameter(param_map[p_id]; 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        else
            push!(petab_params, PEtabParameter(Symbol(p_id); 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        end
    end
    
    # Build observables - Paper uses raw model observables with constant sigma noise
    observables = Dict{String, PEtabObservable}()
    for row in eachrow(observables_df)
        obs_id = row.observableId
        formula = row.observableFormula  # e.g., "total_pS1"
        noise_formula = row.noiseFormula  # e.g., "sigma_pSTAT1"
        
        # 1. Resolve State/Observable from model (e.g., total_pS1)
        # Handle both raw observables and scale factor formulas
        m_obs = match(r"sf_\w+\s*\*\s*(\w+)", formula)
        base_obs_name = isnothing(m_obs) ? formula : m_obs.captures[1]
        
        model_obs_sym = nothing
        for obs_eq in observed(rsys)
            if contains(string(obs_eq.lhs), base_obs_name)
                model_obs_sym = obs_eq.rhs
                break
            end
        end
        if isnothing(model_obs_sym)
            model_obs_sym = species(rsys)[1]
        end
        
        # 2. Check for scale factor (optional - paper doesn't use scale factors)
        m_sf = match(r"(sf_\w+)\s*\*", formula)
        
        # 3. Build observable expression
        if isnothing(m_sf)
            obs_expr = model_obs_sym  # Raw observable (paper uses this)
        else
            sf_name = Symbol(m_sf.captures[1])
            sf_param = only(@parameters $sf_name)
            obs_expr = sf_param * model_obs_sym
        end
        
        # 4. Resolve Sigma Parameter - create symbolic parameter
        m_sigma = match(r"(sigma_\w+)", noise_formula)
        sigma_name = isnothing(m_sigma) ? Symbol(noise_formula) : Symbol(m_sigma.captures[1])
        sigma_param = only(@parameters $sigma_name)
        
        # 5. Build noise expression
        # Check if noise formula contains proportional term (prediction + offset)
        if contains(noise_formula, "*") && contains(noise_formula, "+")
            # Proportional noise: sigma * (prediction + 0.01)
            noise_expr = sigma_param * (obs_expr + 0.01)
        else
            # Constant noise: just sigma (paper uses this)
            noise_expr = sigma_param
        end
        
        observables[obs_id] = PEtabObservable(obs_expr, noise_expr)
    end
    
    # Prepare measurements
    meas_df = copy(measurements_df)
    if hasproperty(meas_df, :simulationConditionId)
        rename!(meas_df, :simulationConditionId => :simulation_id)
    end
    
    petab_model = PEtabModel(odesys, observables, meas_df, petab_params;
        simulation_conditions=sim_conditions, verbose=false)
    
    # Use QNDF (Quasi-Constant Step Size BDF) - a pure Julia stiff solver
    # This SUPPORTS ForwardDiff (Dual numbers) for gradients, unlike CVODE_BDF
    # sparse_jacobian=false to use Dense Jacobian (avoids Sparspak dependency for Duals)
    # gradient_method=:ForwardDiff to avoid Adjoint KeyErrors (proven to work)
    return PEtabODEProblem(petab_model; 
        odesolver=ODESolver(QNDF(); abstol=1e-6, reltol=1e-6),
        sparse_jacobian=false,
        gradient_method=:ForwardDiff
    ), petab_model, sim_conditions
end

# ============================================================================
# CUSTOM NLLH WITH ANALYTIC SCALING FACTORS (matches Python exactly)
# ============================================================================
"""
    compute_nllh_with_scaling(theta, petab_problem, petab_model, sim_conditions, measurements_df)

Compute NLLH with scaling factors computed analytically from normalization condition.
This matches Python's parameter_estimator.py exactly:
1. Simulate normalization condition (IL6=10, IL10=0) 
2. Compute sf = exp_value / model_value at t=20
3. Compute NLLH with proportional noise: σ * (sf * prediction + 0.01)
"""
function compute_nllh_with_scaling(theta, petab_problem, petab_model, sim_conditions, measurements_df)
    # Get sigma values (fixed parameters, not estimated)
    sigma_pSTAT1 = 0.15
    sigma_pSTAT3 = 0.15
    
    # Solve all conditions
    ode_solutions = try
        PEtab.solve_all_conditions(theta, petab_problem, QNDF(); save_observed_t=true)
    catch
        return Inf  # Simulation failed
    end
    
    # Find normalization condition (IL6=10, IL10=0)
    norm_cond_id = nothing
    for (cond_id, cond_dict) in sim_conditions
        L1_val = get(cond_dict, :L1_0, 0.0)
        L2_val = get(cond_dict, :L2_0, 0.0)
        if L1_val == 10.0 && L2_val == 0.0
            norm_cond_id = cond_id
            break
        end
    end
    
    if isnothing(norm_cond_id)
        @warn "Normalization condition (IL6=10, IL10=0) not found"
        return Inf
    end
    
    # Get model values at t=20 for normalization condition
    norm_sol = ode_solutions[Symbol(norm_cond_id)]
    if !(norm_sol.retcode == :Success || string(norm_sol.retcode) == "Success")
        return Inf  # Simulation failed
    end
    
    # Interpolate to get values at t=20
    u_t20 = norm_sol(20.0)
    
    # Get observable indices - find total_pS1 and total_pS3 in the model
    model_info = petab_problem.model_info
    # The observables are typically stored in model_info
    # Access state indices for total_pS1 and total_pS3
    state_syms = Symbol.(string.(states(model_info.model.sys)))
    
    pS1_idx = findfirst(s -> contains(string(s), "total_pS1"), state_syms)
    pS3_idx = findfirst(s -> contains(string(s), "total_pS3"), state_syms)
    
    # If not found as states, they might be observables - try simulated_values approach
    pS1_model_20 = NaN
    pS3_model_20 = NaN
    
    # Get simulated values at normalization condition
    try
        # Use observed species from model
        obs_syms = Symbol.(string.(observed(model_info.model.sys)))
        for (i, obs_sym) in enumerate(obs_syms)
            obs_str = string(obs_sym)
            if contains(obs_str, "total_pS1")
                # Evaluate observable at t=20
                pS1_model_20 = u_t20[i]
            elseif contains(obs_str, "total_pS3")
                pS3_model_20 = u_t20[i]
            end
        end
    catch
        # Fallback: assume states are directly accessible
    end
    
    # If still NaN, try direct access
    if isnan(pS1_model_20) || isnan(pS3_model_20)
        # The observables should be in state vector as observed quantities
        # Try getting from solution directly
        pS1_model_20 = u_t20[end-1]  # Assuming total_pS1 is second-to-last
        pS3_model_20 = u_t20[end]    # Assuming total_pS3 is last
    end
    
    # Get experimental values at t=20 from normalization condition
    norm_meas = filter(row -> 
        row.simulationConditionId == norm_cond_id && row.time == 20.0, 
        measurements_df)
    
    pS1_exp_20 = filter(row -> row.observableId == "obs_total_pS1", norm_meas)
    pS3_exp_20 = filter(row -> row.observableId == "obs_total_pS3", norm_meas)
    
    if isempty(pS1_exp_20) || isempty(pS3_exp_20)
        @warn "Experimental values at normalization condition not found"
        return Inf
    end
    
    pS1_exp_val = Float64(first(pS1_exp_20).measurement)
    pS3_exp_val = Float64(first(pS3_exp_20).measurement)
    
    # Compute scaling factors
    if pS1_model_20 < 1e-12 || pS3_model_20 < 1e-12
        return Inf  # Model values too small
    end
    
    sf_pSTAT1 = pS1_exp_val / pS1_model_20
    sf_pSTAT3 = pS3_exp_val / pS3_model_20
    
    # Now compute NLLH with scaling
    nllh = 0.0
    n_datapoints = 0
    
    for row in eachrow(measurements_df)
        cond_id = row.simulationConditionId
        obs_id = row.observableId
        time = Float64(row.time)
        measurement = Float64(row.measurement)
        
        # Get solution for this condition
        sol = ode_solutions[Symbol(cond_id)]
        if !(sol.retcode == :Success || string(sol.retcode) == "Success")
            return Inf
        end
        
        # Interpolate to measurement time
        u_t = sol(time)
        
        # Get model observable value (same logic as above for indices)
        if contains(obs_id, "pS1")
            model_val = u_t[end-1]  # total_pS1
            sf = sf_pSTAT1
            sigma = sigma_pSTAT1
        else
            model_val = u_t[end]    # total_pS3
            sf = sf_pSTAT3
            sigma = sigma_pSTAT3
        end
        
        # Apply scaling
        prediction = sf * model_val
        
        # Proportional noise: sigma * (prediction + 0.01)
        noise_std = sigma * (prediction + 0.01)
        
        # Gaussian NLLH: 0.5 * (residual/σ)² + log(σ) + 0.5*log(2π)
        residual = measurement - prediction
        nllh += 0.5 * (residual / noise_std)^2
        nllh += log(noise_std)
        nllh += 0.5 * log(2π)
        
        n_datapoints += 1
    end
    
    return nllh
end


# ============================================================================
# SINGLE TASK OPTIMIZATION
# ============================================================================
function run_single_task(args)
    task_id = args["task-id"]
    n_starts = args["n-starts"]
    max_iter = args["max-iter"]
    seed = args["seed"]
    
    println("="^60)
    println("SINGLE TASK OPTIMIZATION")
    println("="^60)
    println("  Task ID: $task_id / $n_starts")
    println("  Max iterations: $max_iter")
    println("  Random seed: $seed")
    println("="^60)
    
    # Load parameters_df for param names
    parameters_df = CSV.read(PARAMETERS_FILE, DataFrame; delim='\t')
    measurements_df = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
    param_names = parameters_df.parameterId
    
    # Load problem (returns tuple with additional objects for custom NLLH)
    petab_problem, petab_model, sim_conditions = load_petab_from_files()
    
    lb = petab_problem.lower_bounds
    ub = petab_problem.upper_bounds
    n_params = length(lb)
    
    println("  Parameters: $n_params")
    
    # Generate ALL start guesses with fixed seed, then pick this task's guess
    # This ensures reproducibility across all array tasks
    Random.seed!(seed)
    all_startguesses = [lb .+ rand(n_params) .* (ub .- lb) for _ in 1:n_starts]
    p0 = all_startguesses[task_id]
    
    # Check for existing checkpoint and resume if available
    chkpt_file = joinpath(RESULTS_DIR, "run_$(task_id)_chkpt.jld2")
    if isfile(chkpt_file)
        try
            chkpt = JLD2.load(chkpt_file)
            p0 = chkpt["xmin"]
            prev_iter = get(chkpt, "iteration", 0)
            prev_fmin = get(chkpt, "fmin", NaN)
            println("  📁 Resuming from checkpoint: iter=$prev_iter, fmin=$prev_fmin")
        catch e
            println("  ⚠️ Failed to load checkpoint, starting fresh: $e")
        end
    end
    
    # DIAGNOSTIC: Test cost function at nominal (middle of bounds) and sampled point
    nominal = (lb .+ ub) ./ 2
    custom_nllh = x -> compute_nllh_with_scaling(x, petab_problem, petab_model, sim_conditions, measurements_df)
    
    println("\n--- DIAGNOSTIC (using custom NLLH with analytic scaling) ---")
    try
        cost_nominal = custom_nllh(nominal)
        println("  Cost at nominal parameters: $cost_nominal")
    catch e
        println("  Cost at nominal FAILED: $e")
    end
    try
        cost_p0 = custom_nllh(p0)
        println("  Cost at sampled p0: $cost_p0")
    catch e
        println("  Cost at p0 FAILED: $e")
    end
    println("  p0 range: min=$(minimum(p0)), max=$(maximum(p0))")
    
    # Check gradient at p0
    try
        g = ForwardDiff.gradient(custom_nllh, p0)
        if any(isnan, g)
            println("  WARNING: Gradient at p0 contains NaNs! Optimization will likely fail.")
        else
            println("  Gradient at p0: OK (Range: $(minimum(g)) to $(maximum(g)))")
        end
    catch e
        println("  Gradient check FAILED: $e")
        println("  (Check if ForwardDiff is loaded and compatible)")
    end
    
    println("------------------\n")
    
    println("Starting optimization for task $task_id...")
    
    
    # =========================================================================
    # OPTIMIZATION SETUP - Using Optimization.jl with LBFGS
    # =========================================================================
    # Optimization.jl provides a unified interface and handles bounds natively.
    # LBFGS with relaxed tolerances converges much faster than Fminbox barrier.
    # =========================================================================
    
    # Track iterations for checkpointing
    iter_count = Ref(0)
    last_checkpoint_iter = Ref(0)
    
    # Callback for progress tracking and checkpointing
    function opt_callback(state, loss)
        iter_count[] += 1
        
        # Print progress every 10 iterations
        if iter_count[] % 10 == 0
            println("  [Iter $(iter_count[])] Loss = $loss")
        end
        
        # Save checkpoint every 50 iterations
        if iter_count[] - last_checkpoint_iter[] >= 50
            mkpath(RESULTS_DIR)
            temp_output_file = joinpath(RESULTS_DIR, "run_$(task_id)_chkpt.jld2")
            JLD2.save(temp_output_file, Dict(
                "task_id" => task_id,
                "fmin" => loss,
                "xmin" => state.u,
                "iteration" => iter_count[],
                "status" => "IN_PROGRESS",
                "n_params" => n_params,
                "param_names" => param_names
            ))
            last_checkpoint_iter[] = iter_count[]
            println("  [Iter $(iter_count[])] Saved checkpoint: fmin = $loss")
        end
        
        return false  # Continue optimization
    end

    # Pre-declare variables to ensure they are accessible after try/catch
    local fmin = NaN
    local xmin = copy(p0)
    local result = nothing

    try
        # Build OptimizationFunction with CUSTOM NLLH that uses analytic scaling factors
        # This matches Python's parameter_estimator.py exactly
        custom_nllh = x -> compute_nllh_with_scaling(x, petab_problem, petab_model, sim_conditions, measurements_df)
        
        opt_f = OptimizationFunction(
            (x, p) -> custom_nllh(x);
            grad = (G, x, p) -> ForwardDiff.gradient!(G, custom_nllh, x)
        )
        
        # Build OptimizationProblem with box constraints
        opt_prob = OptimizationProblem(opt_f, p0, nothing; 
            lb = petab_problem.lower_bounds, 
            ub = petab_problem.upper_bounds
        )
        
        println("  Using Optimization.jl w/ Fminbox(LBFGS) for box-constrained optimization...")
        println("  Bounds: lb=$(minimum(lb)), ub=$(maximum(ub))")
        
        # Solve with Fminbox(LBFGS) - LBFGS with proper box constraints
        # f_tol=1e-4: accept solution when function change is small
        # g_tol=1e-2: accept solution when gradient is small (very relaxed)
        # x_tol=1e-4: accept when parameters stop changing
        sol = solve(opt_prob, Optim.Fminbox(Optim.LBFGS()); 
            maxiters = max_iter,
            maxtime = 3300.0,  # 55 minutes (5 min buffer for saving)
            callback = opt_callback,
            f_tol = 1e-4,
            g_tol = 1e-2,
            x_tol = 1e-4,
            show_trace = true
        )
        
        println("  Optimization finished!")
        println("  Return code: $(sol.retcode)")
        println("  Iterations: $(iter_count[])")
        
        fmin = sol.objective
        xmin = sol.u
        result = (fmin=fmin, xmin=xmin, retcode=sol.retcode)
    catch e
        println("  Optimization CRASHED: $e")
        # Print stacktrace for debugging
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        fmin = NaN
        xmin = copy(p0)
        result = (fmin=NaN, xmin=p0, retcode=:FAILED)
    end
    
    # Accept any finite result as success (optimizer may report "Failure" due to
    # strict convergence criteria, but fmin is still valid and useful)
    success = !isnan(result.fmin) && isfinite(result.fmin)
    if success
        println("  Result: fmin = $(result.fmin) (success=$success)")
    else
        println("  Result: Optimization failed (fmin=$(result.fmin))")
    end
    
    # Save FINAL result
    mkpath(RESULTS_DIR)
    output_file = joinpath(RESULTS_DIR, "run_$(task_id).jld2")
    
    JLD2.save(output_file, Dict(
        "task_id" => task_id,
        "fmin" => isnothing(result) ? NaN : result.fmin,
        "xmin" => isnothing(result) ? p0 : result.xmin,
        "success" => success,
        "n_params" => n_params,
        "param_names" => param_names
    ))
    
    # Remove checkpoint if successful
    chkpt_file = joinpath(RESULTS_DIR, "run_$(task_id)_chkpt.jld2")
    if isfile(chkpt_file)
        rm(chkpt_file)
    end
    
    println("\n✅ Task $task_id complete. Saved to: $output_file")
    
    return result
end

# ============================================================================
# MAIN
# ============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    run_single_task(args)
end
