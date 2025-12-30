# python_compatible_nllh.jl
# Shared utilities to make Julia NLLH match python/parameter_estimator.py

using PEtab
using OrdinaryDiffEq
using ForwardDiff

const _NLLH_DEBUG = get(ENV, "NLLH_DEBUG", "0") != "0"

const _DUALSAFE_SOLVE_LOGGED = Ref(false)
const _DUALSAFE_SOLVE_TYPES_LOGGED = Ref(false)
const _DUALSAFE_NORM_TYPES_LOGGED = Ref(false)
const _DUALSAFE_OBS_TYPES_LOGGED = Ref(false)
const _DUALSAFE_SOLVE_ERROR_LOGGED = Ref(false)
const _DUALSAFE_KEY_MISMATCH_LOGGED = Ref(false)
const _DUALSAFE_COND_ERROR_LOGGED = Ref(false)

@inline _inf_like_type(::Type{T}) where {T} = zero(T) + Inf


struct _NLLHMeasurementCache{M}
    n_meas::Int
    cond_key::Vector{Symbol}         # per measurement row: simulation condition key
    time::Vector{Float64}            # per measurement row
    meas::Vector{Float64}            # per measurement row
    obs_sym::Vector{Symbol}          # per measurement row
    is_s1::BitVector                 # per measurement row (true => pS1)
    maprow::Vector{M}                # per measurement row: xindices.mapxobservable[r]
    time_index::Vector{Int}          # per measurement row: index into saveat vector for that condition
    saveat_by_condition::Dict{Symbol, Vector{Float64}}  # keys include both experiment cid and simulation cid
end

const _NLLH_CACHE = IdDict{Any, Any}()

function _build_nllh_measurement_cache(petab_problem::PEtab.PEtabODEProblem)
    mi = petab_problem.model_info
    df = mi.petab_measurements

    obs_col = _getcol(df, [:observable_id, :observableId])
    sim_col = _getcol(df, [:simulation_condition_id, :simulation_id, :simulationConditionId])
    time_col = _getcol(df, [:time])
    meas_col = _getcol(df, [:measurement])

    n_meas = length(meas_col)

    cond_key = Vector{Symbol}(undef, n_meas)
    time = Vector{Float64}(undef, n_meas)
    meas = Vector{Float64}(undef, n_meas)
    obs_sym = Vector{Symbol}(undef, n_meas)
    is_s1 = BitVector(undef, n_meas)
    mapxobs = mi.xindices.mapxobservable
    M = eltype(mapxobs)
    maprow = Vector{M}(undef, n_meas)

    # Gather measurement rows
    @inbounds for r in 1:n_meas
        cond_key[r] = Symbol(String(sim_col[r]))
        time[r] = Float64(time_col[r])
        meas[r] = Float64(meas_col[r])
        obs_sym[r] = _obsid_symbol(obs_col[r])
        is_s1[r] = occursin("pS1", String(obs_col[r]))
        maprow[r] = mapxobs[r]
    end

    # Build per-simulation-condition saveat vectors (unique sorted measurement times)
    saveat_sim = Dict{Symbol, Vector{Float64}}()
    for r in 1:n_meas
        ck = cond_key[r]
        v = get!(saveat_sim, ck) do
            Float64[]
        end
        push!(v, time[r])
    end
    for (k, v) in saveat_sim
        unique!(v)
        sort!(v)
    end

    # Expand mapping to include experiment condition ids (used for callbacks/tmax lookup during solve)
    saveat_by_condition = Dict{Symbol, Vector{Float64}}()
    siminfo = mi.simulation_info
    for (i, cid) in pairs(siminfo.conditionids[:experiment])
        simid = siminfo.conditionids[:simulation][i]
        simkey = Symbol(simid)
        ckey = Symbol(cid)
        times_sim = get(saveat_sim, simkey, Float64[])
        # Store under both keys
        saveat_by_condition[simkey] = times_sim
        saveat_by_condition[ckey] = times_sim
    end

    # Precompute per-row time index into that condition's saveat vector
    time_index = Vector{Int}(undef, n_meas)
    for r in 1:n_meas
        ck = cond_key[r]
        ts = get(saveat_by_condition, ck, nothing)
        if ts === nothing || isempty(ts)
            # Should never happen if measurements reference valid simulationConditionId
            time_index[r] = 0
            continue
        end
        t = time[r]
        idx = searchsortedfirst(ts, t)
        if idx > length(ts) || ts[idx] != t
            # Penalize time mismatch just like Python does
            time_index[r] = 0
        else
            time_index[r] = idx
        end
    end

    return _NLLHMeasurementCache{M}(n_meas, cond_key, time, meas, obs_sym, is_s1, maprow, time_index, saveat_by_condition)
end

@inline function _get_nllh_cache(petab_problem::PEtab.PEtabODEProblem)
    return get!(_NLLH_CACHE, petab_problem) do
        _build_nllh_measurement_cache(petab_problem)
    end
end

function _solve_all_conditions_dualsafe(theta, petab_problem::PEtab.PEtabODEProblem, osolver;
    abstol::Real = 1e-8,
    reltol::Real = 1e-8,
    maxiters = nothing,
    ntimepoints_save::Integer = 0,
    save_observed_t::Bool = false,
    saveat_by_condition::Union{Nothing, Dict{Symbol, Vector{Float64}}} = nothing,
    dense::Bool = false,
)
    # PEtab's condition solving uses per-condition cached p/u0 buffers (Float64-typed).
    # When differentiating with ForwardDiff.Dual, those caches downcast Dual -> Float64
    # inside `_switch_condition`, killing gradients. Here we bypass those caches and
    # build per-condition p/u0 with the correct element type.

    probinfo = petab_problem.probinfo
    model_info = petab_problem.model_info

    simulation_info = model_info.simulation_info
    model = model_info.model
    xindices = model_info.xindices
    nstates = model_info.nstates

    # Configure solver wrapper
    osolver_wrap = probinfo.solver
    osolver_wrap.abstol = abstol
    osolver_wrap.reltol = reltol
    osolver_wrap.solver = osolver
    if !isnothing(maxiters)
        osolver_wrap.maxiters = maxiters
    end

    # Split & transform dynamic parameters to linear scale (Dual-safe)
    xdyn, _xobs, _xnoise, _xnond = PEtab.split_x(theta, xindices)
    # NOTE: `split_x` returns views (`SubArray`) into `theta`. PEtab.transform_x internally
    # creates a Vector and then tries to `convert(typeof(x), ...)`. If `x` is a SubArray,
    # that convert is impossible (Vector -> SubArray), which breaks ForwardDiff.
    # Materialize to a plain Vector to keep transform_x happy and Dual-safe.
    xdynamic_ps = PEtab.transform_x(collect(xdyn), xindices.xids[:dynamic], xindices; to_xscale = false)

    T = eltype(xdynamic_ps)

    # Base ODEProblem remade to the correct eltype
    base_prob = probinfo.odeproblem
    oprob = remake(base_prob,
        p = convert.(T, base_prob.p),
        u0 = convert.(T, base_prob.u0),
    )

    # Set constant parameters that are shared across conditions
    map_oprob = xindices.map_odeproblem
    @views oprob.p[map_oprob.sys_to_dynamic] .= xdynamic_ps[map_oprob.dynamic_to_sys]

    # No cached solutions here; return a fresh dict
    odesols = Dict{Symbol, Any}()

    # Currently this repo does not use pre-equilibration. If it becomes needed, we can
    # implement it Dual-safe as well.
    if simulation_info.has_pre_equilibration == true
        error("Dual-safe solve path does not yet support pre-equilibration")
    end

    # Solve each experimental condition
    for (i, cid) in pairs(simulation_info.conditionids[:experiment])
        simid = simulation_info.conditionids[:simulation][i]

        local map_cid, p, u0, tmax, oprob_cid, cb, sol
        try
            # Condition-specific parameter mapping
            map_cid = xindices.maps_conidition_id[simid]

            p = copy(oprob.p)
            u0 = similar(oprob.u0, nstates)
            @views u0 .= oprob.u0[1:nstates]

            # Apply condition constant values + condition-specific dynamic values
            p[map_cid.isys_constant_values] .= convert.(T, map_cid.constant_values)
            p[map_cid.ix_sys] .= xdynamic_ps[map_cid.ix_dynamic]

            # Initial state can depend on condition-specific parameters
            model.u0!((@view u0[1:nstates]), p)

            # tspan per condition
            # Important for AD: keep time on Float64. Making `tspan` Dual can propagate Dual time
            # into callbacks/solver internals and trigger MethodError(convert, ...) failures.
            tmax = Float64(simulation_info.tmaxs[cid])
            oprob_cid = remake(oprob, p = p, u0 = u0, tspan = (0.0, tmax))

            # Solve
            cb = simulation_info.callbacks[cid]
            # NOTE: do not pass `saveat = nothing`. Some OrdinaryDiffEq versions call
            # `isempty(saveat)` internally and error on `Nothing`.
            local saveat
            if saveat_by_condition !== nothing
                saveat = get(saveat_by_condition, Symbol(cid), nothing)
                if saveat === nothing
                    saveat = get(saveat_by_condition, Symbol(simid), nothing)
                end
                if saveat === nothing
                    # Fallback: behave like non-saveat call (dense output)
                    sol = solve(oprob_cid, osolver_wrap.solver;
                        abstol = osolver_wrap.abstol,
                        reltol = osolver_wrap.reltol,
                        maxiters = osolver_wrap.maxiters,
                        callback = cb,
                        dense = true,
                        save_everystep = false,
                    )
                else
                    sol = solve(oprob_cid, osolver_wrap.solver;
                        abstol = osolver_wrap.abstol,
                        reltol = osolver_wrap.reltol,
                        maxiters = osolver_wrap.maxiters,
                        callback = cb,
                        saveat = saveat,
                        dense = dense,
                        save_everystep = false,
                    )
                end
            elseif save_observed_t
                sol = solve(oprob_cid, osolver_wrap.solver;
                    abstol = osolver_wrap.abstol,
                    reltol = osolver_wrap.reltol,
                    maxiters = osolver_wrap.maxiters,
                    callback = cb,
                    saveat = Float64.(simulation_info.tsaves[cid]),
                    dense = true,
                    save_everystep = false,
                )
            else
                sol = solve(oprob_cid, osolver_wrap.solver;
                    abstol = osolver_wrap.abstol,
                    reltol = osolver_wrap.reltol,
                    maxiters = osolver_wrap.maxiters,
                    callback = cb,
                    dense = true,
                    save_everystep = false,
                )
            end
        catch err
            if !_DUALSAFE_COND_ERROR_LOGGED[]
                println("[python_compatible_nllh] Dual-safe condition evaluation failed:")
                println("  experiment cid = ", cid)
                println("  simulation  cid = ", simid)
                if @isdefined(oprob_cid)
                    try
                        println("  typeof(oprob_cid.p[1]) = ", typeof(oprob_cid.p[1]))
                        println("  typeof(oprob_cid.u0[1]) = ", typeof(oprob_cid.u0[1]))
                    catch
                    end
                end
                if @isdefined(cb)
                    try
                        println("  callback type = ", typeof(cb))
                    catch
                    end
                end
                println("  error + backtrace = ")
                bt = Base.catch_backtrace()
                showerror(stdout, err, bt)
                println()
                _DUALSAFE_COND_ERROR_LOGGED[] = true
            end
            rethrow()
        end

        if _NLLH_DEBUG && !_DUALSAFE_SOLVE_TYPES_LOGGED[]
            println("[python_compatible_nllh] Dual-safe solve type check:")
            println("  T = ", T)
            println("  typeof(p[1]) = ", typeof(p[1]))
            println("  typeof(u0[1]) = ", typeof(u0[1]))
            println("  typeof(sol.prob.p[1]) = ", typeof(sol.prob.p[1]))
            # sol.u is a vector of state vectors; check both stored and interpolated values.
            println("  typeof(sol.u[end][1]) = ", typeof(sol.u[end][1]))
            local uinterp
            try
                uinterp = sol(20.0)
                println("  typeof(sol(20.0)[1]) = ", typeof(uinterp[1]))
            catch err
                println("  sol(20.0) interpolation not available (dense=false or solver limitation): ", err)
            end

            # Fail-fast signal: if we're differentiating, we MUST retain Duals somewhere.
            # If solve/interpolation returns Float64, ForwardDiff gradients will be identically zero.
            if occursin("ForwardDiff.Dual", string(T))
                if !(typeof(sol.prob.p[1]) <: ForwardDiff.Dual)
                    println("[python_compatible_nllh] WARNING: sol.prob.p eltype is not Dual during AD.")
                end
                if !(typeof(sol.u[end][1]) <: ForwardDiff.Dual)
                    println("[python_compatible_nllh] WARNING: sol.u state eltype is not Dual during AD (solver likely downcasting).")
                end
            end

            _DUALSAFE_SOLVE_TYPES_LOGGED[] = true
        end

        # Store under both the experimental condition id and simulation condition id.
        # PEtab measurement tables typically reference simulation condition ids.
        odesols[Symbol(cid)] = sol
        odesols[Symbol(simid)] = sol
    end

    return odesols
end

@inline function _obsid_symbol(obsid)::Symbol
    if obsid isa Symbol
        return obsid
    end
    s = String(obsid)
    return startswith(s, "obs_") ? Symbol(s) : Symbol("obs_" * s)
end

# --- Small helpers to robustly access PEtab measurement columns ---

function _getcol(df, candidates::Vector{Symbol})
    for c in candidates
        if hasproperty(df, c)
            return getproperty(df, c)
        end
        # DataFrames-style fallback
        try
            return df[!, c]
        catch
        end
    end
    available = try
        propertynames(df)
    catch
        try
            names(df)
        catch
            "<unknown>"
        end
    end
    error("Could not find any of columns $(candidates) in measurement table. Available: $(available)")
end

"""Find measurement row index in PEtab measurement table."""
function find_measurement_row_index(mi::PEtab.ModelInfo, observable_id::AbstractString, simulation_id::AbstractString, t::Real; atol::Real=1e-12)
    df = mi.petab_measurements
    obs_col = _getcol(df, [:observable_id, :observableId])
    sim_col = _getcol(df, [:simulation_condition_id, :simulation_id, :simulationConditionId])
    time_col = _getcol(df, [:time])

    mask = (String.(obs_col) .== String(observable_id)) .& (String.(sim_col) .== String(simulation_id)) .& (abs.(Float64.(time_col) .- Float64(t)) .<= atol)
    return findfirst(mask)
end

"""
Compute NLLH matching python/parameter_estimator.py.

Key behaviors mirrored from Python:
- Scaling factors computed from normalization condition (L1_0=10, L2_0=0) at t=20:
  sf_pSTAT1 = y_exp / y_model, sf_pSTAT3 = y_exp / y_model
- Prediction for each measurement: prediction = sf * model_value
- Noise model: noise_std = sigma * (prediction + 0.01)
- Gaussian NLLH per datum:
  0.5*(residual/noise_std)^2 + log(noise_std) + 0.5*log(2*pi)
- If any simulation fails, returns Inf.

Notes:
- In the current PETab files in this repo, sigma_* are NOT estimable, so Python always uses default 0.15.
  We do the same here.
"""
function compute_nllh_python_compatible(theta, petab_problem::PEtab.PEtabODEProblem, norm_cond_id::AbstractString, r_pS1_norm::Int, r_pS3_norm::Int; sigma_default::Real=0.15)
    mi = petab_problem.model_info
    cache = _get_nllh_cache(petab_problem)
    infT = _inf_like_type(eltype(theta))

    # Solve all conditions (same theta used for all). Use Dual-safe path during AD.
    is_dual = false
    if @isdefined(ForwardDiff)
        is_dual = eltype(theta) <: ForwardDiff.Dual
    else
        # Fallback that doesn't require ForwardDiff to be in scope
        try
            is_dual = occursin("ForwardDiff.Dual", string(eltype(theta)))
        catch
            is_dual = false
        end
    end

    if _NLLH_DEBUG && is_dual && !_DUALSAFE_SOLVE_LOGGED[]
        println("[python_compatible_nllh v2025-12-29] Detected Dual eltype; using cache-free Dual-safe solve path")
        _DUALSAFE_SOLVE_LOGGED[] = true
    end

    ode_solutions = try
        # Always use our explicit per-condition saveat to avoid per-row interpolation overhead.
        # This is both faster and makes timepoint matching exact (like Python's GCD logic).
        _solve_all_conditions_dualsafe(theta, petab_problem, QNDF();
            saveat_by_condition = cache.saveat_by_condition,
            dense = false,
            save_observed_t = false)
    catch err
        if _NLLH_DEBUG && is_dual && !_DUALSAFE_SOLVE_ERROR_LOGGED[]
            println("[python_compatible_nllh] Dual-safe solve threw an error; returning Inf (this causes zero gradients).")
            println("  error + backtrace = ")
            bt = Base.catch_backtrace()
            showerror(stdout, err, bt)
            println()
            _DUALSAFE_SOLVE_ERROR_LOGGED[] = true
        end
        return infT
    end

    # Get normalization condition solution and model values at t=20
    norm_sol = get(ode_solutions, Symbol(norm_cond_id), nothing)
    if norm_sol === nothing
        if _NLLH_DEBUG && is_dual && !_DUALSAFE_KEY_MISMATCH_LOGGED[]
            println("[python_compatible_nllh] Dual-safe key mismatch: could not find norm condition in solutions.")
            println("  norm_cond_id = ", norm_cond_id, " (key = ", Symbol(norm_cond_id), ")")
            println("  available keys sample = ", first(collect(keys(ode_solutions)), min(10, length(ode_solutions))))
            _DUALSAFE_KEY_MISMATCH_LOGGED[] = true
        end
        return infT
    end
    if !(norm_sol.retcode == :Success || string(norm_sol.retcode) == "Success")
        return infT
    end

    # Avoid interpolation: pull u at saved t=20.0
    t20_idx_s1 = cache.time_index[r_pS1_norm]
    t20_idx_s3 = cache.time_index[r_pS3_norm]
    if t20_idx_s1 == 0 || t20_idx_s3 == 0 || t20_idx_s1 != t20_idx_s3
        return infT
    end
    u_t20 = norm_sol.u[t20_idx_s1]

    if _NLLH_DEBUG && is_dual && !_DUALSAFE_NORM_TYPES_LOGGED[]
        println("[python_compatible_nllh] Dual type check (norm condition):")
        println("  eltype(theta) = ", eltype(theta))
        println("  typeof(norm_sol.prob.p[1]) = ", typeof(norm_sol.prob.p[1]))
        println("  typeof(u_t20[1]) = ", typeof(u_t20[1]))
        _DUALSAFE_NORM_TYPES_LOGGED[] = true
    end

    # Observable evaluation: for this model, observables depend only on the state vector `u`.
    # Avoid PEtab.split_x/transform_x here: those pathways can be non-Dual-safe and make
    # ForwardDiff gradients degenerate, while `mi.model.h` (generated observable function)
    # uses only `u[...]` terms.
    Tθ = eltype(theta)
    xobs_ps = Tθ[]
    xnond_ps = Tθ[]

    # Evaluate observables for the normalization rows
    maprow_s1 = cache.maprow[r_pS1_norm]
    maprow_s3 = cache.maprow[r_pS3_norm]

    pS1_model_20 = PEtab._h(u_t20, 20.0, norm_sol.prob.p, xobs_ps, xnond_ps,
        mi.model.h, maprow_s1, cache.obs_sym[r_pS1_norm], mi.petab_parameters.nominal_value)

    pS3_model_20 = PEtab._h(u_t20, 20.0, norm_sol.prob.p, xobs_ps, xnond_ps,
        mi.model.h, maprow_s3, cache.obs_sym[r_pS3_norm], mi.petab_parameters.nominal_value)

    if _NLLH_DEBUG && is_dual && !_DUALSAFE_OBS_TYPES_LOGGED[]
        println("[python_compatible_nllh] Dual type check (observables @ t=20):")
        println("  typeof(pS1_model_20) = ", typeof(pS1_model_20))
        println("  typeof(pS3_model_20) = ", typeof(pS3_model_20))
        _DUALSAFE_OBS_TYPES_LOGGED[] = true
    end

    # Experimental values are the measurements at those same PEtab rows
    pS1_exp_20 = cache.meas[r_pS1_norm]
    pS3_exp_20 = cache.meas[r_pS3_norm]

    if pS1_model_20 <= 1e-12 || pS3_model_20 <= 1e-12
        return infT
    end

    sf_pSTAT1 = pS1_exp_20 / pS1_model_20
    sf_pSTAT3 = pS3_exp_20 / pS3_model_20

    # NLLH over all measurements
    nllh = 0.0
    @inbounds for r in 1:cache.n_meas
        cond_key = cache.cond_key[r]
        sol = get(ode_solutions, cond_key, nothing)
        if sol === nothing
            if _NLLH_DEBUG && is_dual && !_DUALSAFE_KEY_MISMATCH_LOGGED[]
                println("[python_compatible_nllh] Dual-safe key mismatch: missing solution for measurement condition.")
                println("  cond_key = ", cond_key)
                println("  available keys sample = ", first(collect(keys(ode_solutions)), min(10, length(ode_solutions))))
                _DUALSAFE_KEY_MISMATCH_LOGGED[] = true
            end
            return infT
        end
        if !(sol.retcode == :Success || string(sol.retcode) == "Success")
            return infT
        end

        tidx = cache.time_index[r]
        if tidx == 0
            return infT
        end

        t = cache.time[r]
        u_t = sol.u[tidx]

        model_val = PEtab._h(u_t, t, sol.prob.p, xobs_ps, xnond_ps,
            mi.model.h, cache.maprow[r], cache.obs_sym[r], mi.petab_parameters.nominal_value)

        sf = cache.is_s1[r] ? sf_pSTAT1 : sf_pSTAT3
        sigma = sigma_default

        prediction = sf * model_val
        noise_std = sigma * (prediction + 0.01)

        # Guard against invalid noise model values.
        # In Python/numpy, log(negative) yields NaN (no exception). Here we return Inf
        # (Dual-safe) so Optim/line-search can reject this step instead of crashing.
        if !(noise_std > 0)
            return infT
        end

        residual = cache.meas[r] - prediction
        nllh += 0.5 * (residual / noise_std)^2
        nllh += log(noise_std)
        nllh += 0.5 * log(2pi)
    end

    return nllh
end
