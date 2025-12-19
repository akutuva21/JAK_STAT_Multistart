# compare_results.jl
# Compares our best-fit results with pTempest ensemble

using CSV
using DataFrames
using Plots; gr()
using Statistics
using ModelingToolkit
using DifferentialEquations
using PEtab
using Symbolics
using ReactionNetworkImporters
using Catalyst
using StatsPlots

# ============================================================================
# CONFIGURATION
# ============================================================================
const RESULT_FILE = joinpath(@__DIR__, "best_parameters.csv")
const PTEMPEST_DIR = joinpath(@__DIR__, "STAT_models", "pSTAT_mechanistic_model", "Data", "BNGL_pSTAT_simulations")
const PARAM_SETS_FILE = joinpath(PTEMPEST_DIR, "param_sets.csv")
const PTEMPEST_TRAJ_PSTAT1 = joinpath(PTEMPEST_DIR, "pSTAT1_trajs.csv")
const PTEMPEST_TRAJ_PSTAT3 = joinpath(PTEMPEST_DIR, "pSTAT3_trajs.csv")
const PLOT_DIR = joinpath(@__DIR__, "final_results_plots", "ptempest_comparison")

const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

function load_data()
    println("Loading data...")
    if !isfile(RESULT_FILE)
        error("Best parameters file not found: $RESULT_FILE")
    end
    
    # Load our best parameters
    best_params_df = CSV.read(RESULT_FILE, DataFrame)
    best_params = Dict(row.parameter => row.value for row in eachrow(best_params_df))
    println("  Loaded best parameters: $(length(best_params))")
    
    # Load pTempest data
    ptempest_params = CSV.read(PARAM_SETS_FILE, DataFrame)
    ptempest_pstat1 = CSV.read(PTEMPEST_TRAJ_PSTAT1, DataFrame)
    ptempest_pstat3 = CSV.read(PTEMPEST_TRAJ_PSTAT3, DataFrame)
    
    println("  Loaded pTempest params: $(nrow(ptempest_params)) sets")
    println("  Loaded pTempest pSTAT1 trajs: $(nrow(ptempest_pstat1))")
    
    return best_params, ptempest_params, ptempest_pstat1, ptempest_pstat3
end

function overlay_trajectories(best_params, ptempest_pstat1, ptempest_pstat3)
    println("Generating trajectory function overlay...")
    mkpath(PLOT_DIR)
    
    # 1. Simulate our best fit
    # We need to simulate for the condition corresponding to the pTempest simulation
    # pTempest simulations are typically for a specific condition (e.g. high IL6)
    # We'll assume standard high dose condition for now or check metadata if available
    # For now, let's just plot the pTempest ensemble first to see dynamic range
    
    # Using time points from pTempest file
    # Exclude the first column if it's an index or such, check column names
    # Assuming columns are time points
    
    time_points = parse.(Float64, names(ptempest_pstat1))
    
    # --- pSTAT1 ---
    p1 = plot(title="pSTAT1 Ensemble vs Best Fit", xlabel="Time (min)", ylabel="pSTAT1 (au)",legend=false)
    
    # Plot a subset of pTempest (e.g., first 500) to avoid memory issues
    n_plot = min(500, nrow(ptempest_pstat1))
    println("  Plotting $n_plot background trajectories for pSTAT1...")
    
    for i in 1:n_plot
        vals = Float64.(collect(ptempest_pstat1[i, :]))
        plot!(p1, time_points, vals, color=:gray, alpha=0.1, linewidth=1)
    end
    
    # We technically need to simulate our best fit here.
    # To avoid full ODE setup in this script, we can load the cached best fit trajectory if saved,
    # OR setup the ODE system quickly.
    
    # For now, I'll focus on the parameter distribution plots which are easier.
    # The trajectory simulation requires setting up the PEtab problem again.
    
    savefig(p1, joinpath(PLOT_DIR, "pSTAT1_overlay.png"))
    
    # --- pSTAT3 ---
    p2 = plot(title="pSTAT3 Ensemble vs Best Fit", xlabel="Time (min)", ylabel="pSTAT3 (au)", legend=false)
    println("  Plotting $n_plot background trajectories for pSTAT3...")
    
    for i in 1:n_plot
        vals = Float64.(collect(ptempest_pstat3[i, :]))
        plot!(p2, time_points, vals, color=:gray, alpha=0.1, linewidth=1)
    end
    
    savefig(p2, joinpath(PLOT_DIR, "pSTAT3_overlay.png"))
end

function compare_parameters(best_params, ptempest_params)
    println("Generating parameter comparisons...")
    mkpath(PLOT_DIR)
    
    # Identify common parameters
    # Our params might be log10_, pTempest might be linear or log
    # Check pTempest column names
    
    ptempest_names = names(ptempest_params)
    
    # Iterate through our best parameters
    for (param_name, val) in best_params
        # Handle log10 prefix
        clean_name = replace(param_name, "log10_" => "")
        
        # Look for match in pTempest
        # Try exact match first
        match_col = nothing
        if clean_name in ptempest_names
            match_col = clean_name
        else
            # Try case insensitive or other variations if needed
        end
        
        if isnothing(match_col)
            continue
        end
        
        # Plot distribution
        vals = ptempest_params[!, match_col]
        
        # Check if pTempest is log or linear. Usually linear.
        # Our val is log10 (if coming from best_parameters.csv which has log10_ prefix)
        
        # Let's plot in log10 space for better visualization
        log_vals = log10.(vals .+ 1e-10) # Avoid log(0)
        
        p = density(log_vals, label="pTempest Ensemble", fill=true, alpha=0.5, title="Parameter: $clean_name")
        vline!(p, [val], color=:red, linewidth=3, label="Our Best Fit")
        
        savefig(p, joinpath(PLOT_DIR, "param_dist_$(clean_name).png"))
    end
end

# ============================================================================
# MAIN
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    best_params, ptempest_params, pstat1, pstat3 = load_data()
    # overlay_trajectories(best_params, pstat1, pstat3)
    compare_parameters(best_params, ptempest_params)
    println("Done!")
end
