# run_multistart_from_files.jl
# Loads PEtab problem from TSV files, allowing manual editing of parameters/conditions

using Pkg
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, CSV
using Optimization, Optim, OptimizationOptimJL
using SymbolicUtils, Symbolics
using Random
using OrdinaryDiffEq  # For QNDF solver
using ForwardDiff

include(joinpath(@__DIR__, "python_compatible_nllh.jl"))

# ============================================================================
# CONFIGURATION - Edit paths as needed
# ============================================================================
const MODEL_NET = joinpath(@__DIR__, "..", "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "..", "petab_files")

const MEASUREMENTS_FILE = joinpath(PETAB_DIR, "measurements.tsv")
const CONDITIONS_FILE = joinpath(PETAB_DIR, "conditions.tsv")
const PARAMETERS_FILE = joinpath(PETAB_DIR, "parameters.tsv")
const OBSERVABLES_FILE = joinpath(PETAB_DIR, "observables.tsv")

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

function load_petab_from_files()
    println("Loading PEtab problem from TSV files...")
    println("  Model: $MODEL_NET")
    println("  PEtab files: $PETAB_DIR")
    
    # 1. Load the BNGL model
    prn = loadrxnetwork(BNGNetwork(), MODEL_NET)
    rsys = complete(prn.rn)
    odesys = structural_simplify(convert(ODESystem, rsys); simplify=true)
    
    # 2. Load PEtab TSV files
    measurements_df = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
    conditions_df = CSV.read(CONDITIONS_FILE, DataFrame; delim='\t')
    parameters_df = CSV.read(PARAMETERS_FILE, DataFrame; delim='\t')
    observables_df = CSV.read(OBSERVABLES_FILE, DataFrame; delim='\t')
    
    println("  Loaded $(nrow(measurements_df)) measurements")
    println("  Loaded $(nrow(conditions_df)) conditions")
    println("  Loaded $(nrow(parameters_df)) parameters")
    
    # 3. Build simulation conditions dictionary
    model_params = prn.p
    param_map = Dict(string(Symbolics.getname(k)) => k for (k, v) in model_params)
    
    sim_conditions = Dict{String, Dict{Symbol, Float64}}()
    for row in eachrow(conditions_df)
        cond_id = row.conditionId
        cond_dict = Dict{Symbol, Float64}()
        for col in names(conditions_df)
            if col != "conditionId" && !ismissing(row[col])
                # Map column name to model parameter symbol
                if haskey(param_map, col)
                    cond_dict[Symbolics.getname(param_map[col])] = Float64(row[col])
                else
                    cond_dict[Symbol(col)] = Float64(row[col])
                end
            end
        end
        sim_conditions[cond_id] = cond_dict
    end
    
    # 4. Build PEtab parameters list
    petab_params = PEtabParameter[]
    for row in eachrow(parameters_df)
        p_id = row.parameterId
        p_scale = Symbol(row.parameterScale)
        p_lb = Float64(row.lowerBound)
        p_ub = Float64(row.upperBound)
        p_nominal = Float64(row.nominalValue)
        p_estimate = row.estimate == 1
        
        # Check if parameter exists in model or is an observable parameter
        if haskey(param_map, p_id)
            push!(petab_params, PEtabParameter(param_map[p_id]; 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        else
            # Observable parameter (sf_, sigma_, etc.)
            push!(petab_params, PEtabParameter(Symbol(p_id); 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        end
    end
    
    # 5. Build observables dictionary
    # We dynamically identify scale factors and noise parameters from observables_df
    observables = Dict{String, PEtabObservable}()
    
    for row in eachrow(observables_df)
        obs_id = row.observableId
        formula = row.observableFormula
        noise_formula = row.noiseFormula
        
        # 1. Resolve State/Observable from model (e.g., total_pS1)
        # Extract the part after "sf_XYZ * "
        m_obs = match(r"sf_\w+\s*\*\s*(\w+)", formula)
        if isnothing(m_obs)
            @warn "Could not parse observable formula: $formula. Trying fallback."
            base_obs_name = formula # Fallback
        else
            base_obs_name = m_obs.captures[1]
        end
        
        model_obs_sym = nothing
        for obs_eq in observed(rsys)
            obs_lhs_str = string(obs_eq.lhs)
            if contains(obs_lhs_str, base_obs_name)
                model_obs_sym = obs_eq.rhs
                break
            end
        end
        
        if isnothing(model_obs_sym)
            error("Could not find observable '$base_obs_name' in model.")
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
    
    # 6. Prepare measurements DataFrame for PEtab
    # Rename columns to match PEtab.jl expectations
    meas_df = copy(measurements_df)
    if hasproperty(meas_df, :simulationConditionId)
        rename!(meas_df, :simulationConditionId => :simulation_id)
    end
    
    # 7. Create PEtab model and problem
    petab_model = PEtabModel(
        odesys, observables, meas_df, petab_params;
        simulation_conditions=sim_conditions,
        verbose=true
    )
    
    petab_problem = PEtabODEProblem(petab_model;
        odesolver=ODESolver(QNDF(); abstol=1e-6, reltol=1e-6),
        sparse_jacobian=false,
        gradient_method=:ForwardDiff
    )
    
    println("✅ PEtab problem loaded successfully")
    println("  Parameters to estimate: $(length(petab_problem.lower_bounds))")
    
    return petab_problem
end

# ============================================================================
# MULTISTART OPTIMIZATION
# ============================================================================

function run_multistart(n_starts=10; max_iterations=1000)
    petab_problem = load_petab_from_files()

    # Determine normalization condition id and the PEtab row indices used to compute scaling factors.
    # (Matches python/parameter_estimator.py: normalization at L1_0=10, L2_0=0 and t=20)
    mi = petab_problem.model_info

    # Find normalization condition id from the simulation_conditions stored in the PEtab model
    norm_cond_id = nothing
    for (cond_id, cond_dict) in petab_problem.model.simulation_conditions
        L1_val = get(cond_dict, :L1_0, 0.0)
        L2_val = get(cond_dict, :L2_0, 0.0)
        if L1_val == 10.0 && L2_val == 0.0
            norm_cond_id = String(cond_id)
            break
        end
    end
    if isnothing(norm_cond_id)
        error("Could not find normalization condition (L1_0=10, L2_0=0) in conditions.tsv")
    end

    r_pS1_norm = find_measurement_row_index(mi, "obs_total_pS1", norm_cond_id, 20.0)
    r_pS3_norm = find_measurement_row_index(mi, "obs_total_pS3", norm_cond_id, 20.0)
    if isnothing(r_pS1_norm) || isnothing(r_pS3_norm)
        error("Could not locate normalization measurement rows at t=20 for condition $(norm_cond_id)")
    end

    custom_nllh = x -> compute_nllh_python_compatible(x, petab_problem, norm_cond_id, r_pS1_norm, r_pS3_norm)

    opt_f = OptimizationFunction(
        (x, p) -> custom_nllh(x);
        grad = (G, x, p) -> ForwardDiff.gradient!(G, custom_nllh, x)
    )
    
    # Generate start guesses
    lb = petab_problem.lower_bounds
    ub = petab_problem.upper_bounds
    n_params = length(lb)
    
    Random.seed!(1234)
    startguesses = [lb .+ rand(n_params) .* (ub .- lb) for _ in 1:n_starts]
    
    println("\n🚀 Starting multistart optimization")
    println("  Starts: $n_starts")
    println("  Parameters: $n_params")
    println("  Max iterations per start: $max_iterations")
    
    best_res = nothing
    best_f = Inf
    results = []
    
    for (i, p0) in enumerate(startguesses)
        print("Run $i/$n_starts: ")
        try
            opt_prob = OptimizationProblem(opt_f, p0, nothing;
                lb = petab_problem.lower_bounds,
                ub = petab_problem.upper_bounds
            )

            sol = solve(opt_prob, Optim.Fminbox(Optim.LBFGS());
                maxiters = max_iterations,
                show_trace = false
            )

            push!(results, sol)

            if !isnan(sol.objective) && sol.objective < best_f
                best_f = sol.objective
                best_res = sol
                println("✓ NEW BEST fmin=$(round(sol.objective, digits=4))")
            else
                println("fmin=$(isnan(sol.objective) ? "NaN" : round(sol.objective, digits=4))")
            end
        catch e
            println("✗ Failed: $(typeof(e))")
        end
    end
    
    println("\n" * "="^60)
    println("OPTIMIZATION COMPLETE")
    println("="^60)
    
    if !isnothing(best_res) && !isinf(best_f)
        println("Best cost: $best_f")
        println("\nBest parameters:")
        for (i, (name, val)) in enumerate(zip(petab_problem.θ_names, best_res.u))
            println("  $name = $(round(val, digits=6))")
        end
        
        # Save results
        results_file = joinpath(@__DIR__, "optimization_results.csv")
        results_df = DataFrame(parameter = collect(petab_problem.θ_names), value = best_res.u)
        CSV.write(results_file, results_df)
        println("\n💾 Results saved to: $results_file")
    else
        println("⚠️  No successful optimization runs. Check model/data compatibility.")
    end
    
    return results, best_res
end

# ============================================================================
# MAIN
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Default: 5 starts for testing, increase for production runs
    n_starts = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 5
    run_multistart(n_starts)
end
