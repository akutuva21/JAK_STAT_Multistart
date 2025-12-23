# compare_results.jl
# Compares our best-fit results with pTempest ensemble

using CSV
using DataFrames
using Plots; gr()
using Statistics
using PEtab
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using OrdinaryDiffEq
using Symbolics
using SymbolicUtils

# ============================================================================
# CONFIGURATION
# ============================================================================
const RESULT_FILE = joinpath(@__DIR__, "..", "results", "sameaspaper", "best_parameters.csv")
const DATA_DIR = joinpath(@__DIR__, "..", "data")
const PTEMPEST_TRAJ_PSTAT1 = joinpath(DATA_DIR, "pSTAT1_trajs.csv")
const PTEMPEST_TRAJ_PSTAT3 = joinpath(DATA_DIR, "pSTAT3_trajs.csv")
const PTEMPEST_PARAMS = joinpath(DATA_DIR, "param_sets.csv")
const PLOT_DIR = joinpath(@__DIR__, "..", "results", "sameaspaper", "final_results_plots", "ptempest_comparison")

# PEtab/Model files
const MODEL_NET = joinpath(@__DIR__, "..", "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "..", "petab_files")
const MEASUREMENTS_FILE = joinpath(PETAB_DIR, "measurements.tsv")
const CONDITIONS_FILE = joinpath(PETAB_DIR, "conditions.tsv")
const PARAMETERS_FILE = joinpath(PETAB_DIR, "parameters.tsv")
const OBSERVABLES_FILE = joinpath(PETAB_DIR, "observables.tsv")

# Target condition for comparison (IL-6 10 ng/mL, IL-10 = 0)
const TARGET_L1 = 10.0  # IL-6 concentration
const TARGET_L2 = 0.0   # IL-10 concentration

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function load_ptempest_trajectories()
    println("Loading pTempest trajectories...")
    
    # Load parameter sets to identify conditions
    if !isfile(PTEMPEST_PARAMS)
        error("param_sets.csv not found at $PTEMPEST_PARAMS - needed to filter by condition")
    end
    
    param_sets = CSV.read(PTEMPEST_PARAMS, DataFrame)
    println("  Total trajectories in files: $(nrow(param_sets))")
    
    # Show condition breakdown
    println("  Condition breakdown:")
    conditions = combine(groupby(param_sets, [:L1_0, :L2_0]), nrow => :count)
    for row in eachrow(conditions)
        println("    L1=$(row.L1_0), L2=$(row.L2_0): $(row.count) trajectories")
    end
    
    # Filter for target condition (IL-6 10 ng/mL, IL-10 = 0)
    condition_mask = (param_sets.L1_0 .== TARGET_L1) .& (param_sets.L2_0 .== TARGET_L2)
    n_filtered = sum(condition_mask)
    println("  Filtering for L1=$(TARGET_L1), L2=$(TARGET_L2): $(n_filtered) trajectories")
    
    if n_filtered == 0
        error("No trajectories found for target condition L1=$(TARGET_L1), L2=$(TARGET_L2)")
    end
    
    # Load trajectory files (no header)
    pstat1_all = CSV.read(PTEMPEST_TRAJ_PSTAT1, DataFrame; header=false)
    pstat3_all = CSV.read(PTEMPEST_TRAJ_PSTAT3, DataFrame; header=false)
    
    # Apply condition filter
    pstat1_filtered = pstat1_all[condition_mask, :]
    pstat3_filtered = pstat3_all[condition_mask, :]
    
    # Time points: 0 to 90 minutes in 1-minute steps
    n_timepoints = ncol(pstat1_filtered)
    time_points = collect(0.0:(n_timepoints-1))
    
    println("  Loaded $(nrow(pstat1_filtered)) trajectories for IL-6 $(Int(TARGET_L1)) ng/mL condition")
    println("  Time range: 0 to $(n_timepoints-1) minutes")
    
    # NORMALIZE pTempest Data to match our model's normalization (t=20 min = 1.0)
    # Our model is normalized to IL6=10 @ 20min ≈ 1.0
    # pTempest is in absolute units (uM), so we must normalize it to compare DYNAMICS
    
    # Find t=20 index (assuming 1-min steps starting at 0, index 21 is t=20)
    t20_idx = findfirst(==(20.0), time_points)
    if isnothing(t20_idx)
        println("  Warning: t=20 not found in pTempest time points. Using peak for normalization.")
        t20_idx = argmax(vec(median(Matrix(pstat3_filtered), dims=1)))
    end
    
    # Calculate median value at t=20 for normalization
    pS1_vals = Matrix(pstat1_filtered)
    pS3_vals = Matrix(pstat3_filtered)
    
    med_pS1_t20 = median(pS1_vals[:, t20_idx])
    med_pS3_t20 = median(pS3_vals[:, t20_idx])
    
    println("  Normalizing pTempest to median @ t=20:")
    println("    pSTAT1 median @ 20min: $med_pS1_t20 -> 1.0")
    println("    pSTAT3 median @ 20min: $med_pS3_t20 -> 1.0")
    
    # Create normalized DataFrames
    pS1_norm = DataFrame(pS1_vals ./ med_pS1_t20, :auto)
    pS3_norm = DataFrame(pS3_vals ./ med_pS3_t20, :auto)
    
    return time_points, pS1_norm, pS3_norm, n_filtered
end

function load_best_parameters()
    println("Loading best parameters...")
    
    if !isfile(RESULT_FILE)
        error("Best parameters file not found: $RESULT_FILE")
    end
    
    best_params_df = CSV.read(RESULT_FILE, DataFrame)
    
    # Create dict: parameter name -> value (already in log10 scale)
    best_params = Dict{String, Float64}()
    for row in eachrow(best_params_df)
        best_params[row.parameter] = row.value
    end
    
    println("  Loaded $(length(best_params)) parameters")
    
    return best_params, best_params_df
end

function load_petab_problem()
    """
    Load PEtab problem from .net file (same as collect_results.jl)
    """
    println("Loading PEtab problem from .net file...")
    
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
        cond_id = string(row.conditionId)
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
    
    # Build observables dictionary
    observables = Dict{String, PEtabObservable}()
    for row in eachrow(observables_df)
        obs_id = row.observableId
        formula = row.observableFormula
        noise_formula = row.noiseFormula
        
        # Resolve State/Observable from model
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
        
        # 4. Resolve Sigma Parameter
        m_sigma = match(r"(sigma_\w+)", noise_formula)
        sigma_name = isnothing(m_sigma) ? Symbol(noise_formula) : Symbol(m_sigma.captures[1])
        sigma_param = only(@parameters $sigma_name)
        
        # 5. Build noise expression
        if contains(noise_formula, "*") && contains(noise_formula, "+")
            noise_expr = sigma_param * (obs_expr + 0.01)
        else
            noise_expr = sigma_param  # Constant noise (paper uses this)
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
    
    return PEtabODEProblem(petab_model; 
        odesolver=ODESolver(QNDF(); abstol=1e-6, reltol=1e-6),
        sparse_jacobian=false,
        gradient_method=:ForwardDiff
    )
end

function simulate_with_petab(best_params)
    """
    Use PEtab to simulate with best-fit parameters.
    Returns trajectories for the IL6=10, IL10=0 condition.
    """
    println("Simulating with PEtab...")
    
    # Load PEtab problem from .net file
    petab_prob = load_petab_problem()
    
    # Build parameter vector from best fit
    param_names = string.(petab_prob.xnames)
    x_best = zeros(length(param_names))
    
    for (i, name) in enumerate(param_names)
        if haskey(best_params, name)
            x_best[i] = best_params[name]
        else
            # Use nominal from PEtab if not in best fit
            x_best[i] = petab_prob.xnominal[i]
            println("  Warning: Using nominal for $name")
        end
    end
    
    println("  Parameter vector built: $(length(x_best)) parameters")
    
    # Get simulated values using PEtab's built-in function
    # This returns values in measurement table order
    sim_vals = petab_prob.simulated_values(x_best)
    
    # Load measurements to get time/condition info
    measurements = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
    
    # Find the high IL6 condition (IL6=10, IL10=0)
    conditions = CSV.read(CONDITIONS_FILE, DataFrame; delim='\t')
    println("  Available conditions: ", conditions.conditionId)
    
    # Find condition with IL6=10, IL10=0 - look for cond_il6_10
    high_il6_cond = nothing
    for cid in conditions.conditionId
        cid_str = string(cid)
        # Match "cond_il6_10" but NOT "cond_il6_10_il10_X"
        if cid_str == "cond_il6_10" || (occursin("il6_10", cid_str) && !occursin("il10", cid_str))
            high_il6_cond = cid_str
            break
        end
    end
    
    if isnothing(high_il6_cond)
        # Fallback: just use first condition with high IL6
        high_il6_cond = string(conditions.conditionId[1])
        println("  Warning: Could not find IL6=10, IL10=0 condition, using $high_il6_cond")
    end
    
    println("  Using condition: $high_il6_cond")
    
    # Extract trajectories for this condition
    mask_pS1 = (measurements.simulationConditionId .== high_il6_cond) .& 
               (measurements.observableId .== "obs_total_pS1")
    mask_pS3 = (measurements.simulationConditionId .== high_il6_cond) .& 
               (measurements.observableId .== "obs_total_pS3")
    
    times_pS1 = measurements.time[mask_pS1]
    times_pS3 = measurements.time[mask_pS3]
    
    # Get indices into sim_vals
    all_indices = 1:nrow(measurements)
    idx_pS1 = all_indices[mask_pS1]
    idx_pS3 = all_indices[mask_pS3]
    
    # Restore extraction from sim_vals (Accidentally removed in previous edit)
    sim_pS1 = sim_vals[idx_pS1]
    sim_pS3 = sim_vals[idx_pS3]
    
    # ---------------------------------------------------------
    # DEBUG: Check dimensions and values
    # ---------------------------------------------------------
    println("  DEBUG: Extracted simulation results")
    println("    times_pS1 length: $(length(times_pS1))")
    println("    sim_pS1 length: $(length(sim_pS1))")
    
    if isempty(sim_pS1)
        error("Simulation returned no values for pSTAT1!")
    end
    
    # Paper doesn't use scale factors - simulated values are directly the model outputs
    # which are normalized to IL-6 10ng/mL @ t=20
    # For comparison with pTempest, we re-normalize to t=20 exactly
    
    # Find t=20 indices
    idx_t20_pS1 = findfirst(t -> abs(t - 20.0) < 0.1, times_pS1)
    idx_t20_pS3 = findfirst(t -> abs(t - 20.0) < 0.1, times_pS3)
    
    println("  DEBUG: t=20 indices: pS1=$idx_t20_pS1, pS3=$idx_t20_pS3")
    
    # Robust normalization: fall back to max if t=20 is missing or zero
    val_t20_pS1 = !isnothing(idx_t20_pS1) ? sim_pS1[idx_t20_pS1] : 0.0
    val_t20_pS3 = !isnothing(idx_t20_pS3) ? sim_pS3[idx_t20_pS3] : 0.0
    
    norm_factor_pS1 = (val_t20_pS1 > 1e-9) ? val_t20_pS1 : maximum(sim_pS1)
    norm_factor_pS3 = (val_t20_pS3 > 1e-9) ? val_t20_pS3 : maximum(sim_pS3)
    
    # Avoid division by zero
    norm_factor_pS1 = max(norm_factor_pS1, 1e-9)
    norm_factor_pS3 = max(norm_factor_pS3, 1e-9)
    
    raw_pS1 = sim_pS1 ./ norm_factor_pS1
    raw_pS3 = sim_pS3 ./ norm_factor_pS3
    
    println("  Simulated $(length(sim_pS1)) pSTAT1 points, $(length(sim_pS3)) pSTAT3 points")
    println("  Re-normalization factors (model val @ t=20):")
    println("    pS1: $(val_t20_pS1) -> used factor $(norm_factor_pS1)")
    println("    pS3: $(val_t20_pS3) -> used factor $(norm_factor_pS3)")
    println("  Normalized ranges:")
    println("    pSTAT1: $(minimum(raw_pS1)) - $(maximum(raw_pS1))")
    println("    pSTAT3: $(minimum(raw_pS3)) - $(maximum(raw_pS3))")
    
    return times_pS1, raw_pS1, times_pS3, raw_pS3
end

function plot_trajectory_overlay(time_points, ptempest_pstat1, ptempest_pstat3,
                                 times_pS1, best_pstat1, times_pS3, best_pstat3,
                                 n_ptempest)
    println("Generating trajectory overlay plots...")
    mkpath(PLOT_DIR)
    
    # Compute stats for annotations
    ptempest_matrix1 = Matrix(ptempest_pstat1)
    ptempest_matrix3 = Matrix(ptempest_pstat3)
    
    ptempest_peaks1 = [maximum(ptempest_matrix1[i, :]) for i in 1:size(ptempest_matrix1, 1)]
    ptempest_peaks3 = [maximum(ptempest_matrix3[i, :]) for i in 1:size(ptempest_matrix3, 1)]
    
    best_peak1 = maximum(best_pstat1)
    best_peak3 = maximum(best_pstat3)
    
    pctl1 = 100 * mean(ptempest_peaks1 .<= best_peak1)
    pctl3 = 100 * mean(ptempest_peaks3 .<= best_peak3)
    
    # --- pSTAT1 ---
    p1 = plot(title="pSTAT1", 
              xlabel="Time (min)", ylabel="Normalized pSTAT1",
              legend=:topright, size=(800, 600), titlefontsize=12)
    
    # Plot pTempest ensemble (subsample to avoid overplotting)
    n_plot = min(500, nrow(ptempest_pstat1))
    println("  Plotting $n_plot pTempest trajectories for pSTAT1...")
    
    for i in 1:n_plot
        vals = Float64.(collect(ptempest_pstat1[i, :]))
        plot!(p1, time_points, vals, color=:gray, alpha=0.05, linewidth=0.5, label="")
    end
    
    # Compute and plot pTempest median and quantiles
    median_traj1 = vec(median(ptempest_matrix1, dims=1))
    q05_traj1 = vec([quantile(ptempest_matrix1[:, j], 0.05) for j in 1:size(ptempest_matrix1, 2)])
    q95_traj1 = vec([quantile(ptempest_matrix1[:, j], 0.95) for j in 1:size(ptempest_matrix1, 2)])
    
    plot!(p1, time_points, median_traj1, color=:blue, linewidth=2.5, 
          label="pTempest Median")
    plot!(p1, time_points, q05_traj1, color=:blue, linewidth=1.5, linestyle=:dash, 
          label="pTempest 5-95%")
    plot!(p1, time_points, q95_traj1, color=:blue, linewidth=1.5, linestyle=:dash, label="")
    
    # Plot best fit (scatter for discrete time points)
    scatter!(p1, times_pS1, best_pstat1, color=:red, markersize=8, 
             label="Best Fit")
    
    savefig(p1, joinpath(PLOT_DIR, "pSTAT1_trajectory_overlay.png"))
    println("  > Saved: pSTAT1_trajectory_overlay.png")
    
    # --- pSTAT3 ---
    # Determine y-axis limit to show both pTempest and best fit
    ymax_ptempest = maximum(ptempest_peaks3)
    ymax_plot = max(ymax_ptempest * 1.1, best_peak3 * 1.1)
    
    p2 = plot(title="pSTAT3", 
              xlabel="Time (min)", ylabel="Normalized pSTAT3",
              legend=:topright, size=(800, 600), titlefontsize=12,
              ylim=(0, ymax_plot))
    
    println("  Plotting $n_plot pTempest trajectories for pSTAT3...")
    
    for i in 1:n_plot
        vals = Float64.(collect(ptempest_pstat3[i, :]))
        plot!(p2, time_points, vals, color=:gray, alpha=0.05, linewidth=0.5, label="")
    end
    
    # Compute and plot pTempest median and quantiles
    median_traj3 = vec(median(ptempest_matrix3, dims=1))
    q05_traj3 = vec([quantile(ptempest_matrix3[:, j], 0.05) for j in 1:size(ptempest_matrix3, 2)])
    q95_traj3 = vec([quantile(ptempest_matrix3[:, j], 0.95) for j in 1:size(ptempest_matrix3, 2)])
    
    plot!(p2, time_points, median_traj3, color=:blue, linewidth=2.5, 
          label="pTempest Median")
    plot!(p2, time_points, q05_traj3, color=:blue, linewidth=1.5, linestyle=:dash, 
          label="pTempest 5-95%")
    plot!(p2, time_points, q95_traj3, color=:blue, linewidth=1.5, linestyle=:dash, label="")
    
    # Add horizontal line at pTempest max for reference
    hline!(p2, [ymax_ptempest], color=:orange, linewidth=1.5, linestyle=:dot,
           label="pTempest Max")
    
    # Plot best fit
    scatter!(p2, times_pS3, best_pstat3, color=:red, markersize=8, 
             label="Best Fit")
    
    savefig(p2, joinpath(PLOT_DIR, "pSTAT3_trajectory_overlay.png"))
    println("  > Saved: pSTAT3_trajectory_overlay.png")
    
    # --- Combined summary plot ---
    p_combined = plot(p1, p2, layout=(1, 2), size=(1600, 600),
                      plot_title="pTempest Ensemble Comparison (IL-6 10 ng/mL)")
    savefig(p_combined, joinpath(PLOT_DIR, "trajectory_comparison_summary.png"))
    println("  > Saved: trajectory_comparison_summary.png")
    
    return p1, p2
end

function compute_comparison_stats(time_points, ptempest_pstat1, ptempest_pstat3,
                                  times_pS1, best_pstat1, times_pS3, best_pstat3,
                                  n_ptempest)
    println("\n" * "="^70)
    println("COMPARISON STATISTICS")
    println("Condition: IL-6 $(Int(TARGET_L1)) ng/mL, IL-10 = $(Int(TARGET_L2))")
    println("pTempest trajectories: $n_ptempest (from Cheemalavagu et al. 2024)")
    println("="^70)
    
    ptempest_matrix1 = Matrix(ptempest_pstat1)
    ptempest_matrix3 = Matrix(ptempest_pstat3)
    
    # Peak time analysis
    println("\n--- Peak Time Analysis ---")
    
    # pTempest peak times
    ptempest_peak_times1 = [argmax(ptempest_matrix1[i, :]) - 1 for i in 1:size(ptempest_matrix1, 1)]
    ptempest_peak_times3 = [argmax(ptempest_matrix3[i, :]) - 1 for i in 1:size(ptempest_matrix3, 1)]
    
    # Best fit peak time (from discrete points)
    best_peak_idx1 = argmax(best_pstat1)
    best_peak_idx3 = argmax(best_pstat3)
    best_peak_time1 = times_pS1[best_peak_idx1]
    best_peak_time3 = times_pS3[best_peak_idx3]
    
    println("pSTAT1 peak time:")
    println("  pTempest: median=$(median(ptempest_peak_times1)) min, range=$(minimum(ptempest_peak_times1))-$(maximum(ptempest_peak_times1)) min")
    println("  Best fit: $(best_peak_time1) min")
    
    println("pSTAT3 peak time:")
    println("  pTempest: median=$(median(ptempest_peak_times3)) min, range=$(minimum(ptempest_peak_times3))-$(maximum(ptempest_peak_times3)) min")
    println("  Best fit: $(best_peak_time3) min")
    
    # Peak amplitude analysis
    println("\n--- Peak Amplitude Analysis ---")
    
    ptempest_peaks1 = [maximum(ptempest_matrix1[i, :]) for i in 1:size(ptempest_matrix1, 1)]
    ptempest_peaks3 = [maximum(ptempest_matrix3[i, :]) for i in 1:size(ptempest_matrix3, 1)]
    
    best_peak1 = maximum(best_pstat1)
    best_peak3 = maximum(best_pstat3)
    
    println("pSTAT1 peak amplitude:")
    println("  pTempest: median=$(round(median(ptempest_peaks1), sigdigits=3))")
    println("           5th-95th: $(round(quantile(ptempest_peaks1, 0.05), sigdigits=3)) - $(round(quantile(ptempest_peaks1, 0.95), sigdigits=3))")
    println("           min-max:  $(round(minimum(ptempest_peaks1), sigdigits=3)) - $(round(maximum(ptempest_peaks1), sigdigits=3))")
    println("  Best fit: $(round(best_peak1, sigdigits=3))")
    
    println("\npSTAT3 peak amplitude:")
    println("  pTempest: median=$(round(median(ptempest_peaks3), sigdigits=3))")
    println("           5th-95th: $(round(quantile(ptempest_peaks3, 0.05), sigdigits=3)) - $(round(quantile(ptempest_peaks3, 0.95), sigdigits=3))")
    println("           min-max:  $(round(minimum(ptempest_peaks3), sigdigits=3)) - $(round(maximum(ptempest_peaks3), sigdigits=3))")
    println("  Best fit: $(round(best_peak3, sigdigits=3))")
    
    # Where does best fit fall in pTempest distribution?
    println("\n--- Percentile Ranking ---")
    
    pctl1 = 100 * mean(ptempest_peaks1 .<= best_peak1)
    pctl3 = 100 * mean(ptempest_peaks3 .<= best_peak3)
    
    println("Best fit pSTAT1 peak is at $(round(pctl1, digits=1))th percentile of pTempest")
    println("Best fit pSTAT3 peak is at $(round(pctl3, digits=1))th percentile of pTempest")
    
    # Ratio analysis
    println("\n--- Ratio to pTempest Bounds ---")
    
    ratio_to_95th_1 = best_peak1 / quantile(ptempest_peaks1, 0.95)
    ratio_to_max_1 = best_peak1 / maximum(ptempest_peaks1)
    ratio_to_95th_3 = best_peak3 / quantile(ptempest_peaks3, 0.95)
    ratio_to_max_3 = best_peak3 / maximum(ptempest_peaks3)
    
    println("pSTAT1:")
    # FIXED: Replaced 'x' symbol
    println("  Best fit / 95th percentile: $(round(ratio_to_95th_1, digits=2))x")
    println("  Best fit / maximum:         $(round(ratio_to_max_1, digits=2))x")
    if ratio_to_max_1 <= 1.0
        println("  [OK] Within pTempest range")
    else
        println("  [NOTE] Above pTempest range")
    end
    
    println("\npSTAT3:")
    # FIXED: Replaced 'x' symbol
    println("  Best fit / 95th percentile: $(round(ratio_to_95th_3, digits=2))x")
    println("  Best fit / maximum:         $(round(ratio_to_max_3, digits=2))x")
    if ratio_to_max_3 <= 1.0
        println("  [OK] Within pTempest range")
    else
        println("  [NOTE] Above pTempest range by $(round((ratio_to_max_3-1)*100, digits=0))%")
    end
    
    # pSTAT3/pSTAT1 ratio comparison
    println("\n--- pSTAT3/pSTAT1 Ratio Analysis ---")
    
    ptempest_ratios = ptempest_peaks3 ./ ptempest_peaks1
    best_ratio = best_peak3 / best_peak1
    
    println("pTempest pSTAT3/pSTAT1 ratio:")
    println("  median: $(round(median(ptempest_ratios), digits=2))")
    println("  5th-95th: $(round(quantile(ptempest_ratios, 0.05), digits=2)) - $(round(quantile(ptempest_ratios, 0.95), digits=2))")
    println("Best fit pSTAT3/pSTAT1 ratio: $(round(best_ratio, digits=2))")
    # FIXED: Replaced 'x' symbol
    println("Ratio difference: $(round(best_ratio / median(ptempest_ratios), digits=1))x pTempest median")
    
    # Summary assessment
    println("\n" * "="^70)
    println("SUMMARY ASSESSMENT")
    println("="^70)
    # FIXED: Replaced emojis with text
    println("pSTAT1: $(round(pctl1, digits=0))th percentile - ", 
            pctl1 >= 5 && pctl1 <= 95 ? "[GOOD AGREEMENT]" : "[OUTSIDE TYPICAL RANGE]")
    println("pSTAT3: $(round(pctl3, digits=0))th percentile - ",
            pctl3 >= 5 && pctl3 <= 95 ? "[GOOD AGREEMENT]" : "[OUTSIDE TYPICAL RANGE]")
    
    if ratio_to_max_3 > 1.0
        println("\nNote: pSTAT3 is $(round(ratio_to_max_3, digits=1))x the pTempest maximum.")
        println("This may indicate different normalization or model configuration.")
    end
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    println("="^70)
    println("pTEMPEST COMPARISON ANALYSIS")
    println("Reference: Cheemalavagu et al. (2024) Cell Systems 15:37-48")
    println("="^70)
    
    # Load data
    time_points, ptempest_pstat1, ptempest_pstat3, n_ptempest = load_ptempest_trajectories()
    best_params, best_params_df = load_best_parameters()
    
    # Simulate best fit using PEtab
    times_pS1, best_pstat1, times_pS3, best_pstat3 = simulate_with_petab(best_params)
    
    # Generate plots
    plot_trajectory_overlay(time_points, ptempest_pstat1, ptempest_pstat3,
                           times_pS1, best_pstat1, times_pS3, best_pstat3,
                           n_ptempest)
    
    # Compute stats
    compute_comparison_stats(time_points, ptempest_pstat1, ptempest_pstat3,
                            times_pS1, best_pstat1, times_pS3, best_pstat3,
                            n_ptempest)
    
    println("\n" * "="^70)
    println("COMPARISON COMPLETE")
    println("Plots saved to: $PLOT_DIR")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end