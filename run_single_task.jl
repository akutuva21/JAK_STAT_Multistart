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
const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "petab_files")
const RESULTS_DIR = joinpath(@__DIR__, "results")

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
    
    # Build observables - MUST use dynamic scale factor construction for each observable
    observables = Dict{String, PEtabObservable}()
    for row in eachrow(observables_df)
        obs_id = row.observableId
        formula = row.observableFormula
        noise_formula = row.noiseFormula
        
        # 1. Resolve State/Observable from model (e.g., total_pS1)
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
        
        # 2. Resolve Scale Factor (e.g., sf_pSTAT1, sf_pSTAT3) - use @parameters for each
        m_sf = match(r"(sf_\w+)\s*\*", formula)
        
        # 3. Resolve Sigma Parameter - create symbolic parameter
        m_sigma = match(r"(sigma_\w+)", noise_formula)
        sigma_name = isnothing(m_sigma) ? Symbol(noise_formula) : Symbol(m_sigma.captures[1])
        sigma_param = only(@parameters $sigma_name)
        
        # 4. Build observable expression with scale factor
        if isnothing(m_sf)
            obs_expr = model_obs_sym
        else
            # CRITICAL: Create a UNIQUE symbolic parameter for EACH observable's scale factor
            sf_name = Symbol(m_sf.captures[1])
            sf_param = only(@parameters $sf_name)
            obs_expr = sf_param * model_obs_sym
        end
        
        # 5. Build PROPORTIONAL NOISE expression: sigma * (scaled_prediction + 0.01)
        # The +0.01 prevents division by zero when prediction ≈ 0
        # This matches the paper's 15% CV noise model
        noise_expr = sigma_param * (obs_expr + 0.01)
        
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
    )
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
    param_names = parameters_df.parameterId
    
    # Load problem
    petab_problem = load_petab_from_files()
    
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
    println("\n--- DIAGNOSTIC ---")
    try
        cost_nominal = petab_problem.nllh(nominal)
        println("  Cost at nominal parameters: $cost_nominal")
    catch e
        println("  Cost at nominal FAILED: $e")
    end
    try
        cost_p0 = petab_problem.nllh(p0)
        println("  Cost at sampled p0: $cost_p0")
    catch e
        println("  Cost at p0 FAILED: $e")
    end
    println("  p0 range: min=$(minimum(p0)), max=$(maximum(p0))")
    
    # Check gradient at p0
    try
        g = ForwardDiff.gradient(petab_problem.nllh, p0)
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
        # Build OptimizationFunction with gradient from PEtab
        # PEtab provides grad!(G, x) so we wrap it for Optimization.jl's (G, x, p) signature
        opt_f = OptimizationFunction(
            (x, p) -> petab_problem.nllh(x);
            grad = (G, x, p) -> petab_problem.grad!(G, x)
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
