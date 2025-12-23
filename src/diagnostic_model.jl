# diagnostic_model.jl
# Comprehensive diagnostic to understand why model fits are poor
# Run: julia --project="$IL6_HOME/bngl_julia" diagnostic_model.jl

using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, CSV
using Symbolics
using OrdinaryDiffEq
using Printf

# ============================================================================
# CONFIGURATION
# ============================================================================
const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "petab_files")
const BEST_PARAMS_FILE = joinpath(@__DIR__, "best_parameters.csv")

const MEASUREMENTS_FILE = joinpath(PETAB_DIR, "measurements.tsv")
const CONDITIONS_FILE = joinpath(PETAB_DIR, "conditions.tsv")
const PARAMETERS_FILE = joinpath(PETAB_DIR, "parameters.tsv")
const OBSERVABLES_FILE = joinpath(PETAB_DIR, "observables.tsv")

# ============================================================================
# LOAD DATA
# ============================================================================
println("="^70)
println("DIAGNOSTIC: JAK-STAT Model Analysis")
println("="^70)

# Load best parameters
if !isfile(BEST_PARAMS_FILE)
    error("Best parameters file not found: $BEST_PARAMS_FILE")
end

best_params_df = CSV.read(BEST_PARAMS_FILE, DataFrame)
best_params = Dict(row.parameter => row.value for row in eachrow(best_params_df))

println("\n📊 BEST PARAMETERS (log10 scale):")
println("-"^50)
for (k, v) in sort(collect(best_params), by=x->x[1])
    linear_val = 10^v
    @printf("  %-35s = %8.4f (linear: %.2e)\n", k, v, linear_val)
end

# Load PEtab files
measurements_df = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
conditions_df = CSV.read(CONDITIONS_FILE, DataFrame; delim='\t')
parameters_df = CSV.read(PARAMETERS_FILE, DataFrame; delim='\t')
observables_df = CSV.read(OBSERVABLES_FILE, DataFrame; delim='\t')

println("\n📊 DATA SUMMARY:")
println("-"^50)
println("  Measurements: $(nrow(measurements_df))")
println("  Conditions: $(nrow(conditions_df))")
println("  Observables: $(nrow(observables_df))")

# ============================================================================
# CHECK 1: SCALE FACTORS (Paper: NO scale factors used)
# ============================================================================
println("\n" * "="^70)
println("CHECK 1: SCALE FACTORS (Paper Methodology)")
println("="^70)

# Note: Paper doesn't use scale factors - check if they exist in old results
sf_pSTAT1_log = get(best_params, "sf_pSTAT1", NaN)
sf_pSTAT3_log = get(best_params, "sf_pSTAT3", NaN)

if isnan(sf_pSTAT1_log) && isnan(sf_pSTAT3_log)
    println("  ✅ No scale factors found (correct per paper methodology)")
    println("  Paper uses raw model observables (total_pS1, total_pS3) directly")
else
    println("  ⚠️  Scale factors found (OLD workflow):")
    println("  sf_pSTAT1 = 10^$(sf_pSTAT1_log) = $(10^sf_pSTAT1_log)")
    println("  sf_pSTAT3 = 10^$(sf_pSTAT3_log) = $(10^sf_pSTAT3_log)")
    println("  Note: Paper methodology does NOT use scale factors")
end

# Check measurement ranges
pSTAT1_meas = measurements_df[measurements_df.observableId .== "obs_total_pS1", :measurement]
pSTAT3_meas = measurements_df[measurements_df.observableId .== "obs_total_pS3", :measurement]

println("\n  pSTAT1 measurement range: $(minimum(pSTAT1_meas)) to $(maximum(pSTAT1_meas))")
println("  pSTAT3 measurement range: $(minimum(pSTAT3_meas)) to $(maximum(pSTAT3_meas))")

# ============================================================================
# CHECK 2: CONDITIONS - Are IL6/IL10 being set correctly?
# ============================================================================
println("\n" * "="^70)
println("CHECK 2: CONDITIONS (IL6/IL10 concentrations)")
println("="^70)

println("\nConditions from conditions.tsv:")
for row in eachrow(conditions_df)
    l1 = hasproperty(conditions_df, :L1_0) ? row.L1_0 : "MISSING"
    l2 = hasproperty(conditions_df, :L2_0) ? row.L2_0 : "MISSING"
    println("  $(row.conditionId): L1_0=$l1, L2_0=$l2")
end

# Check if L1_0 and L2_0 columns exist
if !hasproperty(conditions_df, :L1_0) || !hasproperty(conditions_df, :L2_0)
    println("\n⚠️  WARNING: L1_0 or L2_0 columns missing from conditions.tsv!")
    println("   This would mean IL6/IL10 concentrations are NOT being varied!")
end

# ============================================================================
# CHECK 3: INITIAL CONCENTRATIONS
# ============================================================================
println("\n" * "="^70)
println("CHECK 3: INITIAL CONCENTRATIONS (from best fit)")
println("="^70)

init_params = ["JAK1_0", "JAK2_0", "GP130_0", "IL6R_0", "IL10R1_0", "IL10R2_0", 
               "S1_0", "S3_0", "PTP1_0", "PTP3_0"]

println("\nInitial concentrations (linear scale):")
for p in init_params
    if haskey(best_params, p)
        log_val = best_params[p]
        lin_val = 10^log_val
        @printf("  %-12s = 10^%7.3f = %12.6f\n", p, log_val, lin_val)
    else
        println("  $p = NOT FOUND")
    end
end

# ============================================================================
# CHECK 4: SIMULATE ODE AND COMPUTE OBSERVABLES
# ============================================================================
println("\n" * "="^70)
println("CHECK 4: ODE SIMULATION")
println("="^70)

# Load model
prn = loadrxnetwork(BNGNetwork(), MODEL_NET)
rsys = complete(prn.rn)
odesys = structural_simplify(convert(ODESystem, rsys); simplify=true)

# Get model parameters and species
model_params = prn.p
model_species = species(rsys)
model_param_names = [string(Symbolics.getname(k)) for (k, v) in model_params]

println("\nModel has $(length(model_species)) species and $(length(model_params)) parameters")

# Find the observable expressions
obs_total_pS1 = nothing
obs_total_pS3 = nothing
for obs_eq in observed(rsys)
    obs_name = string(obs_eq.lhs)
    if contains(obs_name, "total_pS1")
        obs_total_pS1 = obs_eq.rhs
    elseif contains(obs_name, "total_pS3")
        obs_total_pS3 = obs_eq.rhs
    end
end

if isnothing(obs_total_pS1)
    println("⚠️  Could not find total_pS1 observable!")
else
    println("\nObservable total_pS1 found: $obs_total_pS1")
end
if isnothing(obs_total_pS3)
    println("⚠️  Could not find total_pS3 observable!")
else
    println("\nObservable total_pS3 found (truncated)")
end

# Build parameter vector for ODE
param_dict = Dict{Symbol, Float64}()
for (k, v) in model_params
    param_name = string(Symbolics.getname(k))
    if haskey(best_params, param_name)
        # Convert from log10 to linear
        param_dict[Symbolics.getname(k)] = 10^best_params[param_name]
    else
        # Use nominal value from model
        param_dict[Symbolics.getname(k)] = Float64(v)
    end
end

# Set L1_0 and L2_0 to zero initially (will override per condition)
param_dict[:L1_0] = 0.0
param_dict[:L2_0] = 0.0

println("\n" * "-"^50)
println("SIMULATING HIGH-DOSE CONDITIONS")
println("-"^50)

# Test a few conditions
test_conditions = [
    ("IL6=10, IL10=0", 10.0, 0.0),
    ("IL6=0, IL10=10", 0.0, 10.0),
    ("IL6=10, IL10=10", 10.0, 10.0),
    ("IL6=0, IL10=0", 0.0, 0.0),
]

tspan = (0.0, 90.0)
saveat = [0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0]

for (cond_name, il6_conc, il10_conc) in test_conditions
    println("\n--- Condition: $cond_name ---")
    
    # Set ligand concentrations
    test_params = copy(param_dict)
    test_params[:L1_0] = il6_conc
    test_params[:L2_0] = il10_conc
    
    println("  L1_0 (IL6) = $(test_params[:L1_0])")
    println("  L2_0 (IL10) = $(test_params[:L2_0])")
    
    # Build parameter vector in correct order
    param_syms = parameters(odesys)
    p_vec = [get(test_params, Symbolics.getname(p), 1.0) for p in param_syms]
    
    # Initial conditions (all species start at their _0 values or 0)
    u0_vec = zeros(length(model_species))
    
    # Try to set initial conditions from parameters
    state_syms = ModelingToolkit.unknowns(odesys)
    u0_map = Dict{Symbol, Float64}()
    
    for (i, s) in enumerate(state_syms)
        s_name = string(Symbolics.getname(s))
        # Check if there's an _0 parameter for this species
        init_param = s_name * "_0"
        # Remove (t) suffix if present
        init_param = replace(init_param, "(t)" => "")
        
        if haskey(test_params, Symbol(init_param))
            u0_vec[i] = test_params[Symbol(init_param)]
        end
    end
    
    # Create and solve ODE problem
    prob = ODEProblem(odesys, u0_vec, tspan, p_vec)
    
    try
        sol = solve(prob, QNDF(); abstol=1e-8, reltol=1e-8, saveat=saveat)
        
        if sol.retcode == :Success || sol.retcode == ReturnCode.Success
            # Extract observable values
            println("  Simulation successful!")
            println("  Time points: $saveat")
            
            # Find indices for pSTAT species
            state_names = [string(Symbolics.getname(s)) for s in state_syms]
            
            # Sum up species containing "pS1" or "pS3" in their name
            pS1_total = zeros(length(saveat))
            pS3_total = zeros(length(saveat))
            
            for (i, name) in enumerate(state_names)
                if contains(name, "pS1") && !contains(name, "pS3")
                    pS1_total .+= sol[i, :]
                end
                if contains(name, "pS3")
                    pS3_total .+= sol[i, :]
                end
            end
            
            println("\n  Raw model predictions (paper uses these directly):")
            println("  Time    total_pS1    total_pS3")
            for (j, t) in enumerate(saveat)
                @printf("  %4.0f    %10.6f    %10.6f\n", t, pS1_total[j], pS3_total[j])
            end
            
            # Note: Paper doesn't use scale factors, so predictions = raw observables
            println("\n  Note: Paper uses raw observables (no scaling)")
            println("  Data is normalized to IL-6 10ng/mL at t=20")
            
            # Compare to data
            println("\n  Actual measurements for this condition:")
            for row in eachrow(conditions_df)
                l1 = hasproperty(conditions_df, :L1_0) ? row.L1_0 : 0.0
                l2 = hasproperty(conditions_df, :L2_0) ? row.L2_0 : 0.0
                if abs(l1 - il6_conc) < 0.01 && abs(l2 - il10_conc) < 0.01
                    cond_id = row.conditionId
                    cond_meas = measurements_df[measurements_df.simulationConditionId .== cond_id, :]
                    
                    pS1_meas = cond_meas[cond_meas.observableId .== "obs_total_pS1", :]
                    pS3_meas = cond_meas[cond_meas.observableId .== "obs_total_pS3", :]
                    
                    println("  Condition ID: $cond_id")
                    println("  pSTAT1 data: time -> measurement")
                    for r in eachrow(pS1_meas)
                        @printf("    t=%4.0f: %10.6f\n", r.time, r.measurement)
                    end
                    println("  pSTAT3 data: time -> measurement")
                    for r in eachrow(pS3_meas)
                        @printf("    t=%4.0f: %10.6f\n", r.time, r.measurement)
                    end
                    break
                end
            end
            
        else
            println("  ⚠️  Simulation failed: $(sol.retcode)")
        end
    catch e
        println("  ❌ Simulation error: $e")
    end
end

# ============================================================================
# CHECK 5: PETAB INTERNAL DATAFRAME COLUMN NAMES
# ============================================================================
println("\n" * "="^70)
println("CHECK 5: PEtab Internal DataFrame Column Names")
println("="^70)

println("\nThis checks whether collect_results.jl is using correct column names...")

try
    prn2 = loadrxnetwork(BNGNetwork(), MODEL_NET)
    rsys2 = complete(prn2.rn)
    odesys2 = structural_simplify(convert(ODESystem, rsys2); simplify=true)
    
    measurements_df2 = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
    conditions_df2 = CSV.read(CONDITIONS_FILE, DataFrame; delim='\t')
    parameters_df2 = CSV.read(PARAMETERS_FILE, DataFrame; delim='\t')
    observables_df2 = CSV.read(OBSERVABLES_FILE, DataFrame; delim='\t')
    
    model_params2 = prn2.p
    param_map2 = Dict(string(Symbolics.getname(k)) => k for (k, v) in model_params2)
    
    sim_conditions2 = Dict{String, Dict{Symbol, Float64}}()
    for row in eachrow(conditions_df2)
        cond_id = string(row.conditionId)
        cond_dict = Dict{Symbol, Float64}()
        for col in names(conditions_df2)
            if col != "conditionId" && !ismissing(row[col])
                if haskey(param_map2, col)
                    cond_dict[Symbolics.getname(param_map2[col])] = Float64(row[col])
                else
                    cond_dict[Symbol(col)] = Float64(row[col])
                end
            end
        end
        sim_conditions2[cond_id] = cond_dict
    end
    
    petab_params2 = PEtabParameter[]
    for row in eachrow(parameters_df2)
        p_id = row.parameterId
        p_scale = Symbol(row.parameterScale)
        p_lb = Float64(row.lowerBound)
        p_ub = Float64(row.upperBound)
        p_nominal = Float64(row.nominalValue)
        p_estimate = row.estimate == 1
        
        if haskey(param_map2, p_id)
            push!(petab_params2, PEtabParameter(param_map2[p_id]; 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        else
            push!(petab_params2, PEtabParameter(Symbol(p_id); 
                value=p_nominal, estimate=p_estimate, scale=p_scale, lb=p_lb, ub=p_ub))
        end
    end
    
    observables2 = Dict{String, PEtabObservable}()
    for row in eachrow(observables_df2)
        obs_id = row.observableId
        formula = row.observableFormula
        noise_formula = row.noiseFormula
        
        m_obs = match(r"sf_\w+\s*\*\s*(\w+)", formula)
        base_obs_name = isnothing(m_obs) ? formula : m_obs.captures[1]
        
        model_obs_sym = nothing
        for obs_eq in observed(rsys2)
            if contains(string(obs_eq.lhs), base_obs_name)
                model_obs_sym = obs_eq.rhs
                break
            end
        end
        if isnothing(model_obs_sym)
            model_obs_sym = species(rsys2)[1]
        end
        
        m_sf = match(r"(sf_\w+)\s*\*", formula)
        m_sigma = match(r"(sigma_\w+)", noise_formula)
        sigma_name = isnothing(m_sigma) ? Symbol(noise_formula) : Symbol(m_sigma.captures[1])
        sigma_param = only(@parameters $sigma_name)
        
        if isnothing(m_sf)
            obs_expr = model_obs_sym  # Raw observable (paper uses this)
        else
            sf_name = Symbol(m_sf.captures[1])
            sf_param = only(@parameters $sf_name)
            obs_expr = sf_param * model_obs_sym
        end
        
        # Detect noise formula type
        if contains(noise_formula, "*") && contains(noise_formula, "+")
            noise_expr = sigma_param * (obs_expr + 0.01)
        else
            noise_expr = sigma_param  # Constant noise (paper uses this)
        end
        observables2[obs_id] = PEtabObservable(obs_expr, noise_expr)
    end
    
    meas_df2 = copy(measurements_df2)
    if hasproperty(meas_df2, :simulationConditionId)
        rename!(meas_df2, :simulationConditionId => :simulation_id)
    end
    
    petab_model2 = PEtabModel(odesys2, observables2, meas_df2, petab_params2;
        simulation_conditions=sim_conditions2, verbose=false)
    
    petab_problem2 = PEtabODEProblem(petab_model2; 
        odesolver=ODESolver(QNDF(); abstol=1e-6, reltol=1e-6),
        sparse_jacobian=false,
        gradient_method=:ForwardDiff
    )
    
    mi = petab_problem2.model_info
    df_internal = mi.petab_measurements
    
    println("\nPEtab INTERNAL DataFrame columns:")
    for col in names(df_internal)
        println("  - $col")
    end
    
    println("\n🔍 Column access test:")
    
    for (test_name, col_sym) in [
        ("df.observable_id", :observable_id),
        ("df.observableId", :observableId),
        ("df.simulation_id", :simulation_id),
        ("df.simulationConditionId", :simulationConditionId),
        ("df.measurement", :measurement),
        ("df.time", :time)
    ]
        try
            _ = getproperty(df_internal, col_sym)
            println("  ✅ $test_name - EXISTS")
        catch
            println("  ❌ $test_name - DOES NOT EXIST!")
        end
    end
    
    println("\n⚠️  If any snake_case columns don't exist, collect_results.jl will SILENTLY FAIL")
    println("   when computing chi-squared and SSE!")

catch e
    println("  ❌ Could not test PEtab internal DataFrame: $e")
end

# ============================================================================
# SUMMARY
# ============================================================================
println("\n" * "="^70)
println("DIAGNOSTIC SUMMARY")
println("="^70)

println("""

LIKELY ISSUES:

1. DATA-MODEL MISMATCH:
   - Paper normalizes to IL-6 10ng/mL at t=20 (measurement = 1.0 at that point)
   - Model predictions should match this scale
   - If model predictions are ~0.001 but data is ~1.0, parameters may be wrong

2. MODEL NOT PRODUCING pSTAT:
   - Check if JAK/receptor concentrations are too low
   - Check if rate constants are in wrong regime
   - The simulation output above will reveal this

3. NOISE MODEL ARTIFACT:
   - With proportional noise and near-zero predictions,
     the log(σ) term dominates NLLH
   - SSE and χ² are computed differently and don't show this

4. CONDITION MAPPING:
   - Verify L1_0 and L2_0 are actually being passed to ODE
   - If all conditions run with IL6=0, IL10=0, model won't activate

NEXT STEPS:
   - Look at the simulation output above
   - If predictions are 0, check parameter values
   - If predictions are reasonable, check noise formula implementation
""")

println("\n" * "="^70)
println("DIAGNOSTIC COMPLETE")
println("="^70)
