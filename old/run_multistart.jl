using Pkg
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, XLSX
using Optimization, Optim, OptimizationOptimJL
using DiffEqCallbacks
using SymbolicUtils, Symbolics
using ComponentArrays
using SciMLBase, SciMLSensitivity
using Logging
using Sundials
using Random
using QuasiMonteCarlo

# Define paths - relative to this script location
const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")
const DATA_PATH = joinpath(@__DIR__, "Data", "pSTAT1_pooled_data.xlsx")

# Mapping from data columns to conditions
# Assumes suffixes: 01->0.1, 1->1.0, 10->10.0
function parse_condition(col_name)
    # Defaults
    l1_val = 0.0 # IL6
    l2_val = 0.0 # IL10
    
    parts = split(col_name, "_")
    
    i = 1
    while i <= length(parts)
        if parts[i] == "il6"
            val_str = parts[i+1]
            if val_str == "01"
                l1_val = 0.1
            else
                l1_val = parse(Float64, val_str)
            end
            i += 2
        elseif parts[i] == "il10"
             val_str = parts[i+1]
            if val_str == "01"
                l2_val = 0.1
            else
                 l2_val = parse(Float64, val_str)
            end
            i += 2
        else
            i += 1
        end
    end
    
    return l1_val, l2_val
end

function setup_petab_problem()
    println("Setting up PEtab Problem for NehaMultistart...")
    
    # 1. Load BNGL Model
    prn = loadrxnetwork(BNGNetwork(), MODEL_NET)
    rsys = complete(prn.rn)
    
    # 2. Parse Data
    if !isfile(DATA_PATH)
        error("Data file not found at: $DATA_PATH")
    end
    
    xf = XLSX.readxlsx(DATA_PATH)
    sheet = xf[1] # Assuming data is in the first sheet
    df = DataFrame(XLSX.gettable(sheet))
    
    meas_rows = []
    
    # Identify time column (case insensitive)
    time_col = nothing
    for n in names(df)
        if lowercase(n) == "time"
            time_col = n
            break
        end
    end
    
    if isnothing(time_col)
        error("Could not find 'Time' column in data file.")
    end
    
    # Parameters to control
    model_params = prn.p
    param_map = Dict(string(Symbolics.getname(k)) => k for (k, v) in model_params)
    
    l1_sym = param_map["L1_0"]
    l2_sym = param_map["L2_0"]
    
    # Prepare conditions and measurements
    sim_conditions = Dict{String, Dict{Symbol, Float64}}()
    
    # Observable mapping: We are looking at pSTAT1
    obs_id = "obs_total_pS1"
    
    for col in names(df)
        if lowercase(col) == "time" || startswith(col, "Date")
            continue
        end
        
        # Parse condition from column name
        l1_val, l2_val = parse_condition(col)
        
        cond_id = "cond_" * col
        sim_conditions[cond_id] = Dict(
            Symbolics.getname(l1_sym) => l1_val,
            Symbolics.getname(l2_sym) => l2_val
        )
        
        # Add measurements
        for row in eachrow(df)
            t = row[time_col]
            val = row[col]
            if !ismissing(val)
                push!(meas_rows, (
                    simulation_id = cond_id,
                    observableId = obs_id,
                    time = Float64(t),
                    measurement = Float64(val)
                ))
            end
        end
    end
    
    meas_df = DataFrame(meas_rows)
    
    # 3. Define Parameters to Estimate
    petab_params = PEtabParameter[]
    
    # Parameters to EXCLUDE from estimation (fixed or controlled)
    exclude_params = ["L1_0", "L2_0", "SOCS3_0", "SOCS1_0", "Time"]
    
    for (p_sym, val) in model_params
        p_name = string(Symbolics.getname(p_sym))
        if p_name in exclude_params
            continue
        end
        
        # Initial bounds
        lb = 1e-6
        ub = 1e4
        
        push!(petab_params, PEtabParameter(p_sym; value=Float64(val), estimate=true, scale=:log10, lb=lb, ub=ub))
    end
    
    # Scaling factor for pSTAT1 - needs wide range to scale model (~1) to fluorescence (~25000)
    sf_sym = Symbol("sf_pSTAT1")
    push!(petab_params, PEtabParameter(sf_sym; value=1000.0, estimate=true, scale=:log10, lb=1e1, ub=1e6))
    
    # Noise parameter - needs to match scale of fluorescence measurements
    sigma_sym = Symbol("sigma_pSTAT1")
    push!(petab_params, PEtabParameter(sigma_sym; value=1000.0, estimate=true, scale=:log10, lb=1e1, ub=1e5))
    
    # 4. Observables
    observables = Dict{String, PEtabObservable}()
    
    # Find the model system symbolic observable
    model_obs_sym = nothing
    for obs_eq in observed(rsys)
        if string(obs_eq.lhs) == "total_pS1(t)" || string(obs_eq.lhs) == "total_pS1"
             model_obs_sym = obs_eq.rhs
             break
        end
    end
    
    if isnothing(model_obs_sym)
        println("Available observables in model:")
        for obs_eq in observed(rsys)
            println(obs_eq.lhs)
        end
        error("Could not find 'total_pS1' observable in the model.")
    end
    
    @parameters sf_pSTAT1 sigma_pSTAT1
    
    observables[obs_id] = PEtabObservable(
        sf_pSTAT1 * model_obs_sym,
        sigma_pSTAT1
    )

    # 5. Create PEtab Model
    odesys = structural_simplify(convert(ODESystem, rsys); simplify=true)
    
    petab_model = PEtabModel(
        odesys, observables, meas_df, petab_params;
        simulation_conditions=sim_conditions,
        verbose=true
    )
    
    # 6. Create PEtabODEProblem from the model
    petab_problem = PEtabODEProblem(petab_model)
    
    return petab_problem
end

function run_multistart(n_starts=10)
    petab_problem = setup_petab_problem()
    
    # Using Optim.IPNewton
    optimizer = Optim.IPNewton()
    
    # Manual start guess generation
    # PEtab bounds are already in the transformed scale (log10 for log-scaled parameters)
    lb = petab_problem.lower_bounds
    ub = petab_problem.upper_bounds
    n_params = length(lb)
    
    # Generate random start points - sample uniformly between bounds
    Random.seed!(1234)  # For reproducibility
    startguesses = Vector{Vector{Float64}}(undef, n_starts)
    for i in 1:n_starts
        # Bounds are already in correct scale, just sample uniformly
        p0 = lb .+ rand(n_params) .* (ub .- lb)
        startguesses[i] = p0
    end
    
    println("Starting multistart optimization with $n_starts starts...")
    println("Number of parameters to estimate: $n_params")
    
    best_res = nothing
    best_f = Inf
    
    results = []
    
    for (i, p0) in enumerate(startguesses)
        println("Run $i/$n_starts")
        try
            res = calibrate(petab_problem, p0, optimizer; options=Optim.Options(iterations=1000, show_trace=false))
            push!(results, res)
            if res.fmin < best_f
                best_f = res.fmin
                best_res = res
                println("  New best: $best_f")
            else
                println("  Cost: $(res.fmin)")
            end
        catch e
            println("  Run failed: $e")
        end
    end
    
    println("\nOptimization complete.")
    println("Best cost: $best_f")
    if !isnothing(best_res)
        println("Best parameters: $(best_res.xmin)")
    end
    
    return results, best_res
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_multistart(5) # Default 5 starts for test
end
