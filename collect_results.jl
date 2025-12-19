# collect_results.jl
# Collects multistart results and runs full analysis pipeline
# Uses IL6_TGFB src/ functions for visualization, identifiability, profiling

# ============================================================================
# ALL IMPORTS AT TOP LEVEL
# ============================================================================
using JLD2
using DataFrames
using CSV
using Statistics
using Printf
using Plots; gr()
using PEtab
using ComponentArrays
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using OrdinaryDiffEq
using Symbolics
using SymbolicUtils
using Colors
using SciMLBase

# Add IL6_TGFB src directory to LOAD_PATH to import analysis functions
const IL6_SRC_CLUSTER = "/net/dali/home/mscbio/ark426/Research/IL6_TGFB/src"
const IL6_SRC_LOCAL = "C:\\IL6_TGFB\\src"
const IL6_SRC = isdir(IL6_SRC_CLUSTER) ? IL6_SRC_CLUSTER : 
                isdir(IL6_SRC_LOCAL) ? IL6_SRC_LOCAL : nothing

const HAS_IL6_TGFB = !isnothing(IL6_SRC)

if HAS_IL6_TGFB
    if IL6_SRC ∉ LOAD_PATH
        push!(LOAD_PATH, IL6_SRC)
    end
    
    # Import IL6_TGFB analysis modules
    include(joinpath(IL6_SRC, "visualization.jl"))
    include(joinpath(IL6_SRC, "identifiability.jl"))
    include(joinpath(IL6_SRC, "profiling_plot.jl"))
else
    @warn "IL6_TGFB src not found - extended analysis (identifiability, profiling) will be skipped"
end


# ============================================================================
# CONSTANTS
# ============================================================================
const RESULTS_DIR = joinpath(@__DIR__, "results")
const PLOTS_DIR = joinpath(@__DIR__, "final_results_plots")

const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "petab_files")
const MEASUREMENTS_FILE = joinpath(PETAB_DIR, "measurements.tsv")
const CONDITIONS_FILE = joinpath(PETAB_DIR, "conditions.tsv")
const PARAMETERS_FILE = joinpath(PETAB_DIR, "parameters.tsv")
const OBSERVABLES_FILE = joinpath(PETAB_DIR, "observables.tsv")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

"""
    extract_vector(x)

Extract vector data from potentially reconstructed JLD2 types.
"""
function extract_vector(x)
    if x isa AbstractVector{<:Real}
        return collect(Float64, x)
    elseif hasproperty(x, :data)
        return collect(Float64, x.data)
    elseif hasfield(typeof(x), :data)
        return collect(Float64, getfield(x, :data))
    else
        fields = fieldnames(typeof(x))
        if !isempty(fields) && first(fields) == :data
            return collect(Float64, getfield(x, first(fields)))
        end
        error("Cannot extract vector from type: $(typeof(x))")
    end
end

"""
    plot_waterfall(ms_result::MultistartResult; metric=:fmin, title="Waterfall Plot", ylabel="Cost")

Create waterfall plot for a specific metric (fmin/NLLH, SSE, or ChiSq).
"""
function plot_waterfall(ms_result; metric=:fmin, title="Waterfall Plot", ylabel="Cost")
    plot_dir = PLOTS_DIR
    mkpath(plot_dir)
    save_path = joinpath(plot_dir, "waterfall_plot_$(metric).png")
    
    # Extract finite objective values
    values = Float64[]
    for run in ms_result.runs
        try
            val = getfield(run, metric)
            if isfinite(val)
                push!(values, val)
            end
        catch
            # Ignore missing fields
        end
    end
    
    if isempty(values)
        @warn "No finite values found for metric $metric"
        return nothing
    end
    
    # Sort for waterfall effect
    sorted_values = sort(values)
    n_runs = length(sorted_values)
    
    println("Creating waterfall plot for $metric with $n_runs finite runs")
    
    # Determine y-axis scaling
    y_min, y_max = extrema(sorted_values)
    use_log_scale = (y_max / max(1e-10, y_min)) > 100
    
    # Create plot
    plt = plot(
        title = title,
        xlabel = "Ordered optimizer run",
        ylabel = ylabel,
        size = (800, 500),
        dpi = 300,
        legend = false,
        framestyle = :box
    )
    
    # Plot connecting line
    if use_log_scale && minimum(sorted_values) > 0
        plot!(plt, 1:n_runs, sorted_values,
              color = RGBA(0.7, 0.7, 0.7, 0.6),
              linewidth = 1,
              yscale = :log10,
              label = "")
    else
        plot!(plt, 1:n_runs, sorted_values,
              color = RGBA(0.7, 0.7, 0.7, 0.6),
              linewidth = 1,
              label = "")
    end
    
    # Plot individual points with color coding
    for (i, val) in enumerate(sorted_values)
        if i == 1
            point_color = :red
            marker_shape = :star
            marker_size = 10
        elseif i <= 5
            point_color = :orange
            marker_shape = :circle
            marker_size = 8
        else
            point_color = :blue
            marker_shape = :circle
            marker_size = 6
        end
        
        scatter!(plt, [i], [val],
                color = point_color,
                markershape = marker_shape,
                markersize = marker_size,
                markerstrokewidth = 1,
                markerstrokecolor = :black,
                alpha = 0.8,
                label = "")
    end
    
    # Set x-axis ticks
    x_tick_spacing = max(1, n_runs ÷ 10)
    x_ticks = 1:x_tick_spacing:n_runs
    plot!(plt, xticks = x_ticks)
    
    savefig(plt, save_path)
    println("✅ Waterfall plot saved to: $save_path")
    
    return plt
end

"""
    compute_metrics(petab_problem, theta_optim)

Compute NLLH, ChiSq, and SSE for a parameter set.
"""
function compute_metrics(petab_problem, theta_optim)
    # Use PEtab's internal function to get residuals (if available) or compute manually
    
    # We can use the cost function to get NLLH
    nllh = try
        petab_problem.nllh(theta_optim)
    catch
        NaN
    end
    
    # To get SSE/ChiSq, we need individual residuals. 
    ode_solutions = try
         PEtab.solve_all_conditions(theta_optim, petab_problem, QNDF(); save_observed_t=true)
    catch
        nothing
    end
    
    if isnothing(ode_solutions)
        return (nllh=nllh, chisq=NaN, sse=NaN)
    end
    
    mi = petab_problem.model_info
    df = mi.petab_measurements
    
    chisq = 0.0
    sse = 0.0
    
    cache = getfield(petab_problem.probinfo, :cache)
    
    for (sol_id, sol) in ode_solutions
        # Skip failed simulations
        if sol.retcode != :Success && string(sol.retcode) != "Success"
            continue
        end
        
        # Find measurements for this condition
        cond_mask = (String.(hasproperty(df, :simulation_id) ? df.simulation_id : df.simulation_condition_id) .== string(sol_id))
        
        # We need to iterate over each measurement for this condition
        cond_indices = findall(cond_mask)
        
        for idx in cond_indices
            t = df.time[idx]
            obs_id = df.observable_id[idx]
            meas_val = df.measurement[idx]
            
            # Map parameters for this specific observation point
            maprow = mi.xindices.mapxobservable[idx]
            
            # Dynamic states
            xdyn, xobs, xnoise, xnond = PEtab.split_x(theta_optim, mi.xindices)
            
            xobs_ps = PEtab.transform_x(xobs, mi.xindices, :xobservable, cache)
            xnoise_ps = PEtab.transform_x(xnoise, mi.xindices, :xnoise, cache)
            xnond_ps = PEtab.transform_x(xnond, mi.xindices, :xnondynamic, cache)
            
            # Evaluate model at time t
            try
                u_t = sol(t)
                
                # Compute observation h
                h_val = PEtab._h(u_t, t, sol.prob.p, xobs_ps, xnond_ps, 
                            mi.model.h, maprow, obs_id,
                            mi.petab_parameters.nominal_value)
                            
                # Compute sigma
                sigma_val = PEtab._sigma(u_t, t, sol.prob.p, xnoise_ps, xnond_ps, 
                                    mi.model.sigma, maprow, obs_id,
                                    mi.petab_parameters.nominal_value)
                
                # Accumulate
                res = meas_val - h_val
                sse += res^2
                chisq += (res / sigma_val)^2
                
            catch e
                 # Skip point if eval fails
            end
        end
    end
    
    return (nllh=nllh, chisq=chisq, sse=sse)
end

"""
    plot_model_fit(petab_problem, theta_optim, param_names)

Generate time-course plots comparing model predictions to experimental data.
Creates plots for each condition showing measured vs predicted values.
"""
function plot_model_fit(petab_problem, theta_optim, param_names)
    plot_dir = joinpath(pwd(), "final_results_plots")
    mkpath(plot_dir)
    
    println("\nGenerating model fit plots...")
    
    # Load measurement data
    measurements = CSV.read(MEASUREMENTS_FILE, DataFrame; delim='\t')
    conditions = CSV.read(CONDITIONS_FILE, DataFrame; delim='\t')
    
    # Get unique observables and conditions
    observables = unique(measurements.observableId)
    condition_ids = unique(measurements.simulationConditionId)
    
    println("  Observables: $(observables)")
    println("  Conditions: $(length(condition_ids))")
    
    # Create condition lookup for L1_0 and L2_0 values
    cond_lookup = Dict(string(row.conditionId) => row for row in eachrow(conditions))
    
    # Extract sf_pSTAT1 from optimized parameters (scaling factor for observable)
    sf_idx = findfirst(x -> contains(string(x), "sf_pSTAT1"), param_names)
    sf_pSTAT1 = !isnothing(sf_idx) ? 10^theta_optim[sf_idx] : 1.0
    println("  Using sf_pSTAT1 = $sf_pSTAT1")
    
    # Solve all conditions using PEtab infrastructure
    # Note: solve_all_conditions(x, prob, solver) requires 3 args!
    ode_solutions = nothing
    try
        println("  Running ODE simulations...")
        ode_solutions = PEtab.solve_all_conditions(
            theta_optim, 
            petab_problem, 
            QNDF();  # Same solver used in optimization
            save_observed_t=true
        )
        println("  ✅ Simulations complete for $(length(ode_solutions)) conditions")
    catch e
        println("  ⚠️  PEtab.solve_all_conditions failed: $e")
        println("  Falling back to data-only plots...")
    end
    
    for obs_id in observables
        println("  Processing observable: $obs_id")
        
        # Filter measurements for this observable
        obs_meas = filter(row -> row.observableId == obs_id, measurements)
        
        # Create multi-panel plot - 5 columns x 3 rows for 15 conditions
        n_cols = 5
        n_rows = 3
        
        has_predictions = !isnothing(ode_solutions)
        plot_title_str = has_predictions ? "Model Fit: $obs_id" : "Data: $obs_id (sim failed)"
        plt = plot(layout=(n_rows, n_cols), size=(1600, 350*n_rows), dpi=150,
                   plot_title=plot_title_str,
                   left_margin=10Plots.mm,
                   right_margin=5Plots.mm,
                   top_margin=5Plots.mm,
                   bottom_margin=10Plots.mm)
        
        for (idx, cond_id) in enumerate(condition_ids)
            cond_meas = filter(row -> row.simulationConditionId == cond_id, obs_meas)
            
            if isempty(cond_meas)
                continue
            end
            
            # Get time points and measurements
            times = Float64.(cond_meas.time)
            meas_values = Float64.(cond_meas.measurement)
            
            # Get condition parameters
            cond_row = cond_lookup[cond_id]
            L1_0 = hasproperty(cond_row, :L1_0) ? Float64(cond_row.L1_0) : 0.0
            L2_0 = hasproperty(cond_row, :L2_0) ? Float64(cond_row.L2_0) : 0.0
            cond_title = "IL6=$L1_0, IL10=$L2_0"
            
            # Get model predictions if simulations succeeded
            pred_values = Float64[]
            if has_predictions
                cond_sym = Symbol(cond_id)
                if haskey(ode_solutions, cond_sym)
                    sol = ode_solutions[cond_sym]
                    # Check return code - handle both symbol and enum forms
                    sol_success = (sol.retcode == :Success) || 
                                  (string(sol.retcode) == "Success")
                    if sol_success
                        mi = petab_problem.model_info
                        df = mi.petab_measurements
                        
                        for t in times
                            try
                                u_t = sol(t)
                                
                                # Find matching row in petab_measurements
                                obs_col = df.observable_id
                                cond_col = hasproperty(df, :simulation_id) ? df.simulation_id : df.simulation_condition_id
                                time_col = df.time
                                
                                mask = (String.(obs_col) .== String(obs_id)) .& 
                                       (String.(cond_col) .== String(cond_id)) .& 
                                       (abs.(Float64.(time_col) .- t) .< 0.01)
                                r = findfirst(mask)
                                
                                if !isnothing(r)
                                    xdyn, xobs, xnoise, xnond = PEtab.split_x(theta_optim, mi.xindices)
                                    cache = getfield(petab_problem.probinfo, :cache)
                                    xobs_ps = PEtab.transform_x(xobs, mi.xindices, :xobservable, cache)
                                    xnond_ps = PEtab.transform_x(xnond, mi.xindices, :xnondynamic, cache)
                                    
                                    maprow = mi.xindices.mapxobservable[r]
                                    h = PEtab._h(u_t, t, sol.prob.p, xobs_ps, xnond_ps, 
                                                mi.model.h, maprow, obs_col[r], 
                                                mi.petab_parameters.nominal_value)
                                    push!(pred_values, h)
                                end
                            catch
                                # Skip this time point
                            end
                        end
                    end
                end
            end
            
            # Choose color based on IL6 level  
            if L1_0 == 0.0
                marker_color = :blue
            elseif L1_0 <= 0.1
                marker_color = :green
            elseif L1_0 <= 1.0
                marker_color = :red
            else
                marker_color = :purple
            end
            
            # Plot measured data
            scatter!(plt, times, meas_values, 
                    subplot=idx,
                    label="Data",
                    marker=:circle,
                    markersize=6,
                    color=marker_color,
                    title=cond_title,
                    xlabel= (idx > 10) ? "Time (min)" : "",  # Only bottom row
                    ylabel= (mod(idx-1, n_cols) == 0) ? obs_id : "",  # Dynamic ylabel
                    legend=false,  # Remove per-subplot legend
                    titlefontsize=9,
                    guidefontsize=8,
                    tickfontsize=7)
            
            # Plot predictions if available
            if length(pred_values) == length(times)
                ord = sortperm(times)
                plot!(plt, times[ord], pred_values[ord],
                     subplot=idx,
                     label="Model",
                     color=:black,
                     linewidth=2,
                     linestyle=:solid)
            end
        end
        
        save_path = joinpath(plot_dir, "model_fit_$(obs_id).png")
        savefig(plt, save_path)
        println("  ✅ Saved: $save_path")
    end
    
    # Also create a summary plot with select conditions for ALL observables
    for obs_id in observables
        summary_plt = plot(layout=(2, 2), size=(800, 600), dpi=150,
                           plot_title="Data Summary: $obs_id")
        
        select_conds = ["cond_il6_10", "cond_il10_10", "cond_il6_10_il10_10", "cond_il6_1_il10_1"]
        colors_list = [:red, :blue, :purple, :orange]
        
        for (idx, cond_id) in enumerate(select_conds)
            if idx > 4 break end
            
            cond_meas = filter(row -> row.simulationConditionId == cond_id && row.observableId == obs_id, measurements)
            if isempty(cond_meas) continue end
            
            times = Float64.(cond_meas.time)
            meas_values = Float64.(cond_meas.measurement)
            
            scatter!(summary_plt, times, meas_values,
                    subplot=idx,
                    label="Data",
                    marker=:circle,
                    markersize=6,
                    color=colors_list[idx],
                    title=replace(cond_id, "cond_" => ""),
                    xlabel="Time (min)",
                    ylabel=obs_id,
                    legend=:topright)
        end
        
        summary_save_path = joinpath(plot_dir, "model_fit_summary_$(obs_id).png")
        savefig(summary_plt, summary_save_path)
        println("  ✅ Summary plot saved: $summary_save_path")
    end
    
    return nothing
end

"""
    load_petab_problem()

Load the PEtab problem for this project.
"""
function load_petab_problem()
    println("Loading PEtab problem...")
    
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
    # We dynamically identify scale factors and noise parameters from observables_df
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
        
        # 2. Resolve Scale Factor - use @parameters for symbolic
        m_sf = match(r"(sf_\w+)\s*\*", formula)
        
        # 3. Resolve Sigma Parameter - create symbolic parameter
        m_sigma = match(r"(sigma_\w+)", noise_formula)
        sigma_name = isnothing(m_sigma) ? Symbol(noise_formula) : Symbol(m_sigma.captures[1])
        sigma_param = only(@parameters $sigma_name)
        
        # 4. Build observable expression
        if isnothing(m_sf)
            # No scale factor, use model observable directly
            obs_expr = model_obs_sym
        else
            # Create symbolic scale factor parameter
            sf_name = Symbol(m_sf.captures[1])
            sf_param = only(@parameters $sf_name)
            obs_expr = sf_param * model_obs_sym
        end
        
        # 5. Build PROPORTIONAL NOISE expression: sigma * (scaled_prediction + 0.01)
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
    
    return PEtabODEProblem(petab_model; 
        odesolver=ODESolver(QNDF(); abstol=1e-6, reltol=1e-6),
        sparse_jacobian=false,
        gradient_method=:ForwardDiff
    ), petab_model
end

"""
    MultistartRun

Mimics the run structure in PEtabMultistartResult.
"""
struct MultistartRun
    fmin::Float64
    xmin::Vector{Float64}
    chisq::Float64
    sse::Float64
end

"""
    MultistartResult

Mimics PEtabMultistartResult structure for compatibility with plot_waterfall.
"""
struct MultistartResult
    runs::Vector{MultistartRun}
    fmin::Float64
    xmin::Vector{Float64}
    nmultistarts::Int
end

"""
    build_multistart_result(results, param_names, petab_problem)

Construct a PEtabMultistartResult-compatible object from loaded JLD2 results.
Computes additional metrics (SSE, ChiSq).
"""
function build_multistart_result(results, param_names, petab_problem)
    runs = MultistartRun[]
    println("Computing metrics for $(length(results)) runs...")
    
    for (i, r) in enumerate(results)
        xmin = haskey(r, "xmin_vec") ? r["xmin_vec"] : extract_vector(r["xmin"])
        
        # Compute metrics
        metrics = compute_metrics(petab_problem, xmin)
        
        # Log progress
        if i % 10 == 0
            println("  Processed $i/$(length(results)) runs")
        end
        
        push!(runs, MultistartRun(
            Float64(r["fmin"]), 
            Vector{Float64}(xmin),
            metrics.chisq,
            metrics.sse
        ))
    end
    
    best_idx = argmin([r.fmin for r in runs])
    best = runs[best_idx]
    
    return MultistartResult(runs, best.fmin, best.xmin, length(runs))
end

"""
    plot_parameter_distribution(ms_result::MultistartResult, petab_prob::PEtabODEProblem)

Local implementation of parameter distribution plot for our MultistartResult type.
Creates a parallel coordinates plot showing how parameters vary across optimization runs.
"""
function plot_parameter_distribution(ms_result::MultistartResult, petab_prob::PEtabODEProblem)
    plot_dir = joinpath(pwd(), "final_results_plots")
    mkpath(plot_dir)
    save_path = joinpath(plot_dir, "parameter_distribution_plot.png")
    
    # Get parameter names and bounds from PEtab
    param_names = string.(petab_prob.xnames)
    n_params = length(param_names)
    lower_bounds = collect(petab_prob.lower_bounds)
    upper_bounds = collect(petab_prob.upper_bounds)
    
    println("Generating parameter distribution plot for $n_params parameters...")
    
    # Collect all parameter estimates (each run.xmin is a Vector{Float64})
    all_estimates = [run.xmin for run in ms_result.runs if length(run.xmin) == n_params]
    best_x = ms_result.xmin
    
    if isempty(all_estimates)
        @warn "No valid parameter estimates to plot"
        return nothing
    end
    
    # Create plot with generous margins to avoid clipping parameter names
    plot_height = max(500, n_params * 40)
    plt = plot(
        title="Parameter Distribution ($(length(all_estimates)) runs)",
        xlabel="Parameter Value (log10)",
        ylabel="",  # Remove ylabel since parameter names are on y-axis
        legend=:topright,
        yticks=(1:n_params, param_names),
        yflip=true,
        framestyle=:box,
        size=(1100, plot_height),
        dpi=300,
        left_margin=25Plots.mm,  # Generous left margin for long parameter names
        right_margin=10Plots.mm,
        bottom_margin=10Plots.mm,
        top_margin=5Plots.mm,
        tickfontsize=8
    )
    
    y_values = 1:n_params
    
    # Plot all runs in gray
    for x_vec in all_estimates
        if x_vec != best_x
            plot!(plt, x_vec, y_values, seriestype=:path, color=:gray, alpha=0.3, linewidth=1, label="")
        end
    end
    
    # Plot bounds
    bounds_y = vcat(y_values, y_values)
    bounds_x = vcat(lower_bounds, upper_bounds)
    scatter!(plt, bounds_x, bounds_y, marker=:+, color=:black, markersize=4, label="Bounds")
    
    # Plot best fit in red
    if length(best_x) == n_params
        plot!(plt, best_x, y_values, 
              seriestype=:path, 
              color=:red, 
              alpha=0.9,
              linewidth=2.5,
              marker=:circle,
              markersize=4,
              label="Best Fit")
    end
    
    savefig(plt, save_path)
    println("✅ Parameter distribution plot saved to: $save_path")
    return plt
end

# ============================================================================
# MAIN COLLECTION FUNCTION
# ============================================================================

function collect_results(; run_ident::Bool=true, run_profiles::Bool=false)
    println("="^60)
    println("COLLECTING MULTISTART RESULTS")
    println("="^60)
    println("Results directory: $RESULTS_DIR")
    
    # Create output directories
    mkpath(PLOTS_DIR)
    
    # Find all result files (exclude checkpoints)
    result_files = filter(f -> endswith(f, ".jld2") && !contains(f, "_chkpt"), readdir(RESULTS_DIR))
    
    if isempty(result_files)
        println("No result files found!")
        return nothing
    end
    
    println("Found $(length(result_files)) result files")
    
    # Load all results
    results = []
    param_names = nothing
    
    for file in result_files
        filepath = joinpath(RESULTS_DIR, file)
        try
            data = JLD2.load(filepath)
            
            # Extract xmin as plain vector to avoid ComponentArray issues
            if haskey(data, "xmin")
                data["xmin_vec"] = extract_vector(data["xmin"])
            end
            
            push!(results, data)
            if isnothing(param_names)
                param_names = data["param_names"]
            end
        catch e
            println("  Warning: Could not load $file: $e")
        end
    end
    
    println("Successfully loaded $(length(results)) results")
    
    # Filter successful runs
    successful = filter(r -> r["success"] && !isnan(r["fmin"]), results)
    println("Successful optimization runs: $(length(successful))")
    
    if isempty(successful)
        println("⚠️  No successful runs found!")
        return nothing
    end
    
    # Load PEtab problem ONCE to get correct parameter names/order
    petab_problem, _ = load_petab_problem()
    correct_param_names = string.(petab_problem.xnames)
    
    # Find best result
    best_idx = argmin([r["fmin"] for r in successful])
    best = successful[best_idx]
    best_xmin = best["xmin_vec"]
    
    println("\n" * "="^60)
    println("BEST RESULT")
    println("="^60)
    println("  Task ID: $(best["task_id"])")
    println("  Cost (fmin): $(best["fmin"])")
    println("\nBest parameters:")
    
    for (name, val) in zip(correct_param_names, best_xmin)
        display_name = replace(string(name), "log10_" => "")
        @printf("  %-35s = %10.6f\n", display_name, val)
    end
    
    # Save summary CSV
    summary_df = DataFrame(
        task_id = [r["task_id"] for r in results],
        fmin = [r["fmin"] for r in results],
        success = [r["success"] for r in results]
    )
    sort!(summary_df, :fmin)
    
    summary_file = joinpath(@__DIR__, "optimization_summary.csv")
    CSV.write(summary_file, summary_df)
    println("\n📊 Summary saved to: $summary_file")
    
    # Save best parameters
    best_params_df = DataFrame(
        parameter = correct_param_names,
        value = collect(best_xmin)
    )
    best_params_file = joinpath(@__DIR__, "best_parameters.csv")
    CSV.write(best_params_file, best_params_df)
    println("💾 Best parameters saved to: $best_params_file")
    
    # Statistics
    valid_fmin = filter(!isnan, [r["fmin"] for r in results])
    println("\n📈 Statistics:")
    println("  Min fmin: $(minimum(valid_fmin))")
    println("  Median fmin: $(median(valid_fmin))")
    println("  Mean fmin: $(mean(valid_fmin))")
    println("  Max fmin: $(maximum(valid_fmin))")
    
    # =========================================================================
    # WATERFALL PLOTS - NLLH, SSE, ChiSq
    # =========================================================================
    println("\n" * "="^60)
    println("GENERATING WATERFALL PLOTS")
    println("="^60)
    
    ms_result = nothing
    try
        # ms_result uses correct names via petab_problem inside build_multistart_result
        ms_result = build_multistart_result(successful, correct_param_names, petab_problem)
        
        # Plot NLLH
        plot_waterfall(ms_result, metric=:fmin, title="Waterfall Plot (NLLH)", ylabel="Negative Log-Likelihood")
        
        # Plot ChiSq
        plot_waterfall(ms_result, metric=:chisq, title="Waterfall Plot (Chi-Squared)", ylabel="Chi-Squared")
        
        # Plot SSE
        plot_waterfall(ms_result, metric=:sse, title="Waterfall Plot (SSE)", ylabel="Sum of Squared Errors")
        
    catch e
        println("⚠️  Could not generate waterfall plots: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end

    # =========================================================================
    # PARAMETER DISTRIBUTION PLOT
    # =========================================================================
    println("\n" * "="^60)
    println("GENERATING PARAMETER DISTRIBUTION PLOT")
    println("="^60)
    
    try
        # Load PEtab problem for plot_parameter_distribution
        petab_problem, _ = load_petab_problem()
        # ms_result was built in the previous section
        plot_parameter_distribution(ms_result, petab_problem)
        println("✅ Parameter distribution plot saved to: $(PLOTS_DIR)/parameter_distribution_plot.png")
    catch e
        println("⚠️  Parameter distribution plotting failed: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
    
    # =========================================================================
    # IDENTIFIABILITY ANALYSIS - Using identifiability.jl
    # =========================================================================
    if run_ident && HAS_IL6_TGFB
        println("\n" * "="^60)
        println("RUNNING IDENTIFIABILITY ANALYSIS (FIM)")
        println("="^60)
        
        try
            petab_problem, petab_model = load_petab_problem()
            
            # Convert best parameters to ComponentVector
            # Use keys from best_xmin to handle fixed parameters correctly
            xmin_param_names = if best_xmin isa ComponentVector
                [replace(string(k), "log10_" => "") for k in keys(best_xmin)]
            else
                param_names[1:length(best_xmin)]
            end
            θ_best = ComponentVector(NamedTuple(zip(Symbol.(xmin_param_names), collect(best_xmin))))
            
            # Run FIM-based identifiability
            ident_result = run_identifiability(petab_problem, θ_best)
            
            if !isnothing(ident_result)
                println("✅ Identifiability analysis complete")
            end
        catch e
            println("⚠️  Identifiability analysis failed: $e")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
    end
    
    # =========================================================================
    # MODEL FIT VISUALIZATION
    # =========================================================================
    println("\n" * "="^60)
    println("GENERATING MODEL FIT PLOTS")
    println("="^60)
    
    try
        petab_problem, petab_model = load_petab_problem()
        θ_best = Vector{Float64}(best_xmin)
        plot_model_fit(petab_problem, θ_best, correct_param_names)
        println("✅ Model fit plots complete")
    catch e
        println("⚠️  Model fit plotting failed: $e")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
    
    # =========================================================================
    # LIKELIHOOD PROFILING - Using profiling_plot.jl (optional, slow)
    # =========================================================================
    if run_profiles && HAS_IL6_TGFB
        println("\n" * "="^60)
        println("RUNNING LIKELIHOOD PROFILING")
        println("="^60)
        
        try
            petab_problem, petab_model = load_petab_problem()
            θ_best = ComponentVector(NamedTuple(zip(Symbol.(param_names), best_xmin)))
            
            run_likelihood_profiling(
                petab_model,
                nothing,
                nothing,
                θ_best,
                Dict{String, Float64}()
            )
            
            println("✅ Likelihood profiling complete")
        catch e
            println("⚠️  Likelihood profiling failed: $e")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
    end
    
    # Save best fit for downstream use
    best_fit_file = joinpath(@__DIR__, "best_fit.jld2")
    JLD2.save(best_fit_file, Dict(
        "task_id" => best["task_id"],
        "fmin" => best["fmin"],
        "xmin" => best_xmin,
        "param_names" => correct_param_names,
        "n_successful" => length(successful),
        "n_total" => length(results)
    ))
    println("\n💾 Best fit saved to: $best_fit_file")
    
    println("\n" * "="^60)
    println("COLLATION COMPLETE")
    println("="^60)
    
    return best
end

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    run_ident = "--ident" in ARGS || "-i" in ARGS
    run_profiles = "--profile" in ARGS || "-p" in ARGS
    
    # Default: run identifiability but not profiling
    if isempty(ARGS)
        run_ident = true
        run_profiles = false
    end
    
    collect_results(; run_ident=run_ident, run_profiles=run_profiles)
end
