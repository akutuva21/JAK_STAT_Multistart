# Script to export and inspect PEtab problem components
# UPDATED: Now averages across BOTH pSTAT1 and pSTAT3 files and ALL sheets
# Follows methodology from: STAT_models/pSTAT_mechanistic_model (Method details)
using Pkg
using ReactionNetworkImporters, Catalyst
using DifferentialEquations, ModelingToolkit
using PEtab, DataFrames, XLSX, CSV
using SymbolicUtils, Symbolics
using Statistics

const MODEL_NET = joinpath(@__DIR__, "..", "variable_JAK_STAT_SOCS_degrad_model.net")
const DATA_PSTAT1 = joinpath(@__DIR__, "..", "Data", "pSTAT1_pooled_data.xlsx")
const DATA_PSTAT3 = joinpath(@__DIR__, "..", "Data", "pSTAT3_pooled_data.xlsx")

function parse_condition(col_name)
    l1_val = 0.0
    l2_val = 0.0
    parts = split(col_name, "_")
    i = 1
    while i <= length(parts)
        if parts[i] == "il6"
            val_str = parts[i+1]
            l1_val = val_str == "01" ? 0.1 : parse(Float64, val_str)
            i += 2
        elseif parts[i] == "il10"
            val_str = parts[i+1]
            l2_val = val_str == "01" ? 0.1 : parse(Float64, val_str)
            i += 2
        else
            i += 1
        end
    end
    return l1_val, l2_val
end

"""
Load all sheets from an Excel file, shift each to start at 0 at t=0,
interpolate and average across all experiments.
"""
function load_and_average_stat(data_path::String, stat_name::String, normalize_per_exp::Bool=false)
    println("Processing $stat_name from: $(basename(data_path))")
    
    xf = XLSX.readxlsx(data_path)
    sheet_names = XLSX.sheetnames(xf)
    println("  Found $(length(sheet_names)) experiments (sheets).")
    
    # data_by_exp: exp_name -> condition -> time -> value
    data_by_exp = Dict{String, Dict{String, Dict{Float64, Float64}}}()
    all_times = Set{Float64}()
    all_conds = Set{String}()
    
    for sheet_name in sheet_names
        sheet = xf[sheet_name]
        df = DataFrame(XLSX.gettable(sheet))
        
        # Find time column
        time_col = nothing
        for n in names(df)
            if lowercase(n) == "time"
                time_col = n
                break
            end
        end
        if isnothing(time_col) continue end
        
        cond_cols = [c for c in names(df) if lowercase(c) != "time" && !startswith(c, "Date")]
        exp_data = Dict{String, Dict{Float64, Float64}}()
        
        # Get baselines (t=0)
        baselines = Dict{String, Float64}()
        for col in cond_cols
            for row in eachrow(df)
                if row[time_col] == 0 && !ismissing(row[col])
                    baselines[col] = Float64(row[col])
                    break
                end
            end
        end
        
        for col in cond_cols
            push!(all_conds, col)
            exp_data[col] = Dict{Float64, Float64}()
            for row in eachrow(df)
                t = Float64(row[time_col])
                val = row[col]
                if !ismissing(val)
                    push!(all_times, t)
                    bg = get(baselines, col, 0.0)
                    shifted = max(0.0, Float64(val) - bg)
                    exp_data[col][t] = shifted
                end
            end
        end
        
        # NORMALIZE PER EXPERIMENT Logic
        if normalize_per_exp
            # Find normalization factor: il6_10 at t=20.0 (approx)
            # Column name usually "il6_10"
            norm_factor = 1.0
            found_norm = false
            
            # Check for column "il6_10" or similar
            norm_col_candidate = "il6_10"
            if haskey(exp_data, norm_col_candidate)
                # Check for t=20.0
                if haskey(exp_data[norm_col_candidate], 20.0)
                   norm_factor = exp_data[norm_col_candidate][20.0]
                   found_norm = true
                end
            end
            
            if found_norm && norm_factor > 0
                println("    Normalizing $sheet_name by factor $norm_factor")
                for (c, t_dict) in exp_data
                    for (t, v) in t_dict
                        t_dict[t] = v / norm_factor
                    end
                end
            else
                println("    WARNING: No normalization factor found for $sheet_name (il6_10 @ 20.0). Using 1.0.")
            end
        end

        data_by_exp[sheet_name] = exp_data
    end
    
    # Compute the overall grid of unique timepoints across ALL experiments
    global_time_grid = sort(collect(all_times))
    
    # Compute averages: condition -> time -> mean
    # NOTE: Only use actual measured values, NO interpolation
    avg_data = Dict{String, Dict{Float64, Float64}}()
    
    for cond in all_conds
        avg_data[cond] = Dict{Float64, Float64}()
        for t in global_time_grid
            vals_at_t = Float64[]
            for (exp, cond_dict) in data_by_exp
                if haskey(cond_dict, cond) && haskey(cond_dict[cond], t)
                    val = cond_dict[cond][t]
                    push!(vals_at_t, val)
                    # DEBUG: Print contributors for specific condition and time
                    if cond == "il6_10_il10_10" && t == 30.0
                        println("  DEBUG: $exp contributes to $cond at t=$t with val=$val")
                    end
                end
            end
            if !isempty(vals_at_t)
                avg_data[cond][t] = mean(vals_at_t)
            end
        end
    end
    
    return avg_data
end

function export_petab_tables()
    println("\n=== Generating Integrated PEtab Tables (pSTAT1 + pSTAT3) ===")
    
    # 1. Load Model
    prn = loadrxnetwork(BNGNetwork(), MODEL_NET)
    rsys = complete(prn.rn)
    
    # 2. Load and Average Data Independently
    # NOW: Normalizing PER EXPERIMENT inside load_and_average_stat to be correct
    avg_pS1 = load_and_average_stat(DATA_PSTAT1, "pSTAT1", true)
    avg_pS3 = load_and_average_stat(DATA_PSTAT3, "pSTAT3", true)
    
    # 3. Filter to 6 conditions shown in Fig 2B
    fig2b_conditions = [
        (10.0, 0.0), (1.0, 0.0), (0.0, 10.0), (0.0, 1.0), (10.0, 10.0), (1.0, 1.0)
    ]
    
    function filter_to_fig2b(avg_data)
        filtered = Dict{String, Dict{Float64, Float64}}()
        for (cond, data) in avg_data
            cond_tuple = parse_condition(cond)
            if cond_tuple in fig2b_conditions
                filtered[cond] = data
            end
        end
        return filtered
    end
    
    avg_pS1 = filter_to_fig2b(avg_pS1)
    avg_pS3 = filter_to_fig2b(avg_pS3)
    
    # 4. Build Combined Measurements Table
    meas_rows = []
    
    # Process pSTAT1
    for (cond, data) in avg_pS1
        cond_id = "cond_" * cond
        for (t, val) in data
            push!(meas_rows, (
                observableId = "obs_total_pS1",
                simulationConditionId = cond_id,
                time = t,
                measurement = val, # Already normalized
                noiseParameters = "sigma_pSTAT1"
            ))
        end
    end
    
    # Process pSTAT3
    for (cond, data) in avg_pS3
        cond_id = "cond_" * cond
        for (t, val) in data
            push!(meas_rows, (
                observableId = "obs_total_pS3",
                simulationConditionId = cond_id,
                time = t,
                measurement = val, # Already normalized
                noiseParameters = "sigma_pSTAT3"
            ))
        end
    end
    
    measurements_df = DataFrame(meas_rows)
    sort!(measurements_df, [:simulationConditionId, :observableId, :time])
    
    # 5. Build Conditions Table (Union)
    all_cond_names = unique(vcat(collect(keys(avg_pS1)), collect(keys(avg_pS3))))
    conditions_rows = []
    for cond in all_cond_names
        l1, l2 = parse_condition(cond)
        push!(conditions_rows, (conditionId = "cond_" * cond, L1_0 = l1, L2_0 = l2))
    end
    conditions_df = DataFrame(conditions_rows)
    
    # 6. Build Observables Table
    # Paper uses raw model observables WITHOUT scale factors
    # Noise: constant sigma (15% CV)
    observables_df = DataFrame(
        observableId = ["obs_total_pS1", "obs_total_pS3"],
        observableFormula = ["total_pS1", "total_pS3"],
        noiseFormula = ["sigma_pSTAT1", "sigma_pSTAT3"]
    )
    
    # 7. Build Parameters Table
    param_rows = []
    model_params = prn.p
    exclude_params = ["L1_0", "L2_0", "SOCS3_0", "SOCS1_0"]  # Only exclude: ligands (experimental) and induced proteins (start at 0)
    
    # Add model parameters to estimate
    for (p_sym, val) in model_params
        p_name = string(Symbolics.getname(p_sym))
        if p_name in exclude_params continue end
        push!(param_rows, (
            parameterId = p_name, parameterScale = "log10",
            lowerBound = 1e-6, upperBound = 1e4,
            nominalValue = Float64(val), estimate = 1
        ))
    end
    
    # Add Sigma parameters (Fixed at 0.15 per paper's 15% CV)
    # NO scale factors - paper doesn't use them
    push!(param_rows, (parameterId="sigma_pSTAT1", parameterScale="log10", lowerBound=1e-2, upperBound=1.0, nominalValue=0.15, estimate=0))
    push!(param_rows, (parameterId="sigma_pSTAT3", parameterScale="log10", lowerBound=1e-2, upperBound=1.0, nominalValue=0.15, estimate=0))
    
    parameters_df = DataFrame(param_rows)
    
    # 8. Export
    output_dir = joinpath(@__DIR__, "..", "petab_files")
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "measurements.tsv"), measurements_df; delim='\t')
    CSV.write(joinpath(output_dir, "conditions.tsv"), conditions_df; delim='\t')
    CSV.write(joinpath(output_dir, "parameters.tsv"), parameters_df; delim='\t')
    CSV.write(joinpath(output_dir, "observables.tsv"), observables_df; delim='\t')
    
    println("\n✅ Integration Complete!")
    println("  Total Measurements: $(nrow(measurements_df))")
    println("  Conditions: $(nrow(conditions_df))")
    println("  Files saved to: $output_dir")
end

export_petab_tables()
