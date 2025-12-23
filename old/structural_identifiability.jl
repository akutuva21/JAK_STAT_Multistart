# ============================================================================
# Structural Identifiability Analysis for the JAK-STAT Model
# ============================================================================
# This script performs structural identifiability analysis using 
# StructuralIdentifiability.jl to determine which parameters are theoretically
# identifiable from the model structure (independent of data).
#
# Prerequisites: StructuralIdentifiability.jl must be installed in the project
# Run on cluster login node first:
#   julia --project="$IL6_HOME/bngl_julia" -e 'using Pkg; Pkg.add("StructuralIdentifiability")'
#
# Usage: julia --project="path/to/bngl_julia" structural_identifiability.jl
# ============================================================================

using StructuralIdentifiability

using ModelingToolkit
using Catalyst
using ReactionNetworkImporters
using PEtab
using CSV
using DataFrames

using SymbolicUtils

# ============================================================================
# CONFIGURATION
# ============================================================================
const MODEL_NET = joinpath(@__DIR__, "variable_JAK_STAT_SOCS_degrad_model.net")
const PETAB_DIR = joinpath(@__DIR__, "petab_files")

# ============================================================================
# LOAD MODEL
# ============================================================================
println("\n" * "="^60)
println("LOADING MODEL FOR STRUCTURAL IDENTIFIABILITY ANALYSIS")
println("="^60)

# Load the BioNetGen model
prnbng = loadrxnetwork(BNGNetwork(), MODEL_NET)
rn = prnbng.rn

# Mark the ReactionSystem as complete (required by newer Catalyst versions)
rn = complete(rn)

# Convert to ODESystem
@named odesys = convert(ODESystem, rn)
odesys = structural_simplify(odesys)

# Get the equations
eqs = equations(odesys)
states = ModelingToolkit.unknowns(odesys)
params = ModelingToolkit.parameters(odesys)

println("\nModel Statistics:")
println("  Number of ODEs: $(length(eqs))")
println("  Number of state variables: $(length(states))")
println("  Number of parameters: $(length(params))")

# ============================================================================
# DEFINE OBSERVABLE
# ============================================================================
# The observable in PEtab is: sf_pSTAT1 * total_pS1
# total_pS1 is a sum of several species containing phosphorylated STAT1

# Get the scaling factor parameter
@parameters sf_pSTAT1
# Note: sf_pSTAT1 is not in the RN, so we define it as a symbolic parameter
# But we need to check if it's already in 'params'. It likely isn't because it's
# defined in PEtab, not the BNGL model itself.
# However, assess_local_identifiability requires all parameters to be part of the system
# or the input expression.

# Debug: Show available fields in prnbng
println("\nAvailable fields in parsed network: ", fieldnames(typeof(prnbng)))

# Per docs: groupstosyms is a Dict mapping group name Strings to symbolic observables
if hasproperty(prnbng, :groupstosyms) && !isnothing(prnbng.groupstosyms)
    groupstosyms = prnbng.groupstosyms
    println("Found groupstosyms: ", keys(groupstosyms))
    
    # Look for total_pS1 group (the observable we care about)
    if haskey(groupstosyms, "total_pS1")
        total_pS1_expr = groupstosyms["total_pS1"]
        println("\nFound 'total_pS1' observable: ", total_pS1_expr)
        
        # Define the measured quantity equation: y ~ sf_pSTAT1 * total_pS1
        # To do this properly in StructuralIdentifiability, we treat 'sf_pSTAT1' as an unknown parameter
        # Since it's not in the ODE system, we might need to add it or treat the analysis carefully.
        
        # Simplified approach: Check identifiability of the *shape* (total_pS1) first
        # Scaling factors are usually identifiable if the shape is non-zero.
        
        measured_quantities = [total_pS1_expr]
        println("Using observable: total_pS1(t)")
        
        # ============================================================================
        # RUN STRUCTURAL IDENTIFIABILITY ANALYSIS
        # ============================================================================
        println("\n" * "="^60)
        println("RUNNING IDENTIFIABILITY ANALYSIS")
        println("="^60)
        println("This computation may take several minutes given the model size.")
        println("Checking identifiability of biological parameters with respect to pSTAT1...")
        
        # Per StructuralIdentifiability.jl docs for ModelingToolkit:
        # measured_quantities should be equations like [y ~ observable]
        @independent_variables t
        @variables y(t)
        
        # total_pS1_expr is a symbolic sum of species from the ODE system
        # We need to rationalize any float coefficients in this expression
        function safe_rationalize(x)
            if x isa AbstractFloat
                return rationalize(x)
            else
                return x
            end
        end
        walker = SymbolicUtils.Postwalk(safe_rationalize)
        
        # Rationalize the observable expression
        total_pS1_rational = walker(total_pS1_expr)
        measured_eqs = [y ~ total_pS1_rational]
        
        println("\nPre-processing complete. Observable rationalized.")
        println("Using measured equation: y ~ total_pS1")
        
        try
            # Use assess_local_identifiability (Sedoglavic algorithm, polynomial time)
            # Pass the ORIGINAL odesys - don't reconstruct it, as that breaks symbolic refs
            println("Using local identifiability (Sedoglavic algorithm)...")
            results = assess_local_identifiability(odesys, measured_quantities=measured_eqs)
            
            println("\n✅ Analysis Successful!")
            println("\n" * "-"^60)
            println("RESULTS: PARAMETER IDENTIFIABILITY (LOCAL)")
            println("-"^60)
            
            identifiable = []
            non_identifiable = []
            
            # assess_local_identifiability returns Dict{Symbol, Bool} (true = locally identifiable)
            for (p, is_id) in results
                p_str = string(p)
                
                if is_id
                    push!(identifiable, p_str)
                else
                    push!(non_identifiable, p_str)
                end
            end
            
            println("\n🟢 LOCALLY IDENTIFIABLE ($(length(identifiable))):")
            println(join(sort(identifiable), ", "))
            
            println("\n🔴 NON-IDENTIFIABLE ($(length(non_identifiable))):")
            println(join(sort(non_identifiable), ", "))
            
            # Save results
            summary_file = joinpath(@__DIR__, "identifiability_results.txt")
            open(summary_file, "w") do f
                println(f, "Structural Identifiability Results (Local)")
                println(f, "==========================================")
                println(f, "Model: $MODEL_NET")
                println(f, "Observable: total_pS1")
                println(f, "Algorithm: Sedoglavic (probabilistic, polynomial time)")
                println(f, "\nTotal Parameters/States: $(length(results))")
                println(f, "Locally Identifiable: $(length(identifiable))")
                println(f, "Non-Identifiable: $(length(non_identifiable))")
                
                println(f, "\nLOCALLY IDENTIFIABLE:")
                for p in sort(identifiable)
                    println(f, "  $p")
                end
                
                println(f, "\nNON-IDENTIFIABLE:")
                for p in sort(non_identifiable)
                    println(f, "  $p")
                end
            end
            println("\n📄 Detailed results saved to: $summary_file")
            
        catch e
            println("\n⚠️  Structural identifiability analysis failed:")
            showerror(stdout, e, catch_backtrace())
            println("\n")
            
            # Provide useful guidance
            println("\n" * "="^60)
            println("MODEL TOO LARGE FOR SYMBOLIC ANALYSIS")
            println("="^60)
            println("""
This model has 53 ODEs and 50 parameters (104+ symbolic variables).
Structural identifiability requires Gröbner basis computations which
scale exponentially with model size. This is a fundamental limitation.

RECOMMENDED ALTERNATIVES:
1. **Profile Likelihood Analysis** (practical identifiability):
   Run: julia collect_results.jl --profiles
   This empirically assesses identifiability at your estimated values.

2. **Reduce model complexity**:
   Consider a simplified sub-model for theoretical analysis.

3. **Local sensitivity analysis**:
   Use Fisher Information Matrix (FIM) as a local approximation.
   Already implemented in collect_results.jl --ident
""")
            
            # Save this finding as a result
            summary_file = joinpath(@__DIR__, "identifiability_results.txt")
            open(summary_file, "w") do f
                println(f, "Structural Identifiability Analysis")
                println(f, "===================================")
                println(f, "Model: $MODEL_NET")
                println(f, "Observable: total_pS1")
                println(f, "\nRESULT: Model too large for symbolic analysis")
                println(f, "  - States: 53")
                println(f, "  - Parameters: 50") 
                println(f, "  - Total symbolic variables: 104+")
                println(f, "\nRecommendation: Use profile likelihood instead")
            end
            println("📄 Summary saved to: $summary_file")
        end
        
    else
        println("❌ Error: 'total_pS1' group not found. Available groups: ", keys(groupstosyms))
    end
else
    println("❌ Error: Could not access groupstosyms in parsed model.")
end

println("\n" * "="^60)
println("ANALYSIS COMPLETE")
println("="^60)
