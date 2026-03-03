#! format: off

# Advanced RASP algorithms for SymbolicRegression.jl
# Builds on the primitives from rasp_operators.jl, adding seq_mul and seq_eq
# to enable prefix sums, pattern detection, and positional arithmetic.

using SymbolicRegression
using DynamicExpressions: GenericOperatorEnum
using MLJBase: machine, fit!, report, MLJBase
using Random
using Random: AbstractRNG
using Test

# ── 1. RaspValue type (same as rasp_operators.jl) ─────────────────────────────

struct RaspValue
    tag::Symbol
    seq::Vector{Float64}
    sel::Matrix{Bool}
end

RaspValue(seq::Vector{Float64}) = RaspValue(:seq, seq, Matrix{Bool}(undef, 0, 0))
RaspValue(sel::Matrix{Bool}) = RaspValue(:sel, Float64[], sel)

sentinel_seq(n::Int) = RaspValue(:seq, fill(NaN, n), Matrix{Bool}(undef, 0, 0))
sentinel_sel(n::Int) = RaspValue(:sel, Float64[], fill(false, n, n))

function _infer_n(a::RaspValue, b::RaspValue)
    a.tag == :seq && length(a.seq) > 1 && return length(a.seq)
    b.tag == :seq && length(b.seq) > 1 && return length(b.seq)
    a.tag == :sel && return size(a.sel, 1)
    b.tag == :sel && return size(b.sel, 1)
    a.tag == :seq && return length(a.seq)
    b.tag == :seq && return length(b.seq)
    return 1
end
_infer_n(a::RaspValue) = a.tag == :seq ? length(a.seq) : size(a.sel, 1)

function _broadcast_seqs(a::Vector{Float64}, b::Vector{Float64})
    la, lb = length(a), length(b)
    la == lb && return (a, b)
    la == 1 && return (fill(a[1], lb), b)
    lb == 1 && return (a, fill(b[1], la))
    n = max(la, lb)
    return (fill(NaN, n), fill(NaN, n))
end

Base.:(==)(a::RaspValue, b::RaspValue) = a.tag == b.tag && (a.tag == :seq ? a.seq == b.seq : a.sel == b.sel)
Base.hash(a::RaspValue, h::UInt) = a.tag == :seq ? hash(a.seq, hash(a.tag, h)) : hash(a.sel, hash(a.tag, h))
Base.isnan(a::RaspValue) = a.tag == :seq && any(isnan, a.seq)
Base.copy(a::RaspValue) = RaspValue(a.tag, copy(a.seq), copy(a.sel))

# ── 2. Operators ───────────────────────────────────────────────────────────────

# --- Selectors (binary: seq, seq → sel) ---

function select_eq(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    sel = Matrix{Bool}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n; sel[i, j] = ks[j] == qs[i]; end
    RaspValue(sel)
end

function select_lt(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    sel = Matrix{Bool}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n; sel[i, j] = ks[j] < qs[i]; end
    RaspValue(sel)
end

function select_gt(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    sel = Matrix{Bool}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n; sel[i, j] = ks[j] > qs[i]; end
    RaspValue(sel)
end

function select_leq(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    sel = Matrix{Bool}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n; sel[i, j] = ks[j] <= qs[i]; end
    RaspValue(sel)
end

function select_true(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    RaspValue(fill(true, length(ks), length(ks)))
end

# --- Unary: SelectorWidth (sel → seq) ---

function selector_width(x::RaspValue)::RaspValue
    x.tag != :sel && return sentinel_seq(_infer_n(x))
    n = size(x.sel, 1)
    result = Vector{Float64}(undef, n)
    @inbounds for i in 1:n; result[i] = Float64(count(x.sel[i, :])); end
    RaspValue(result)
end

# --- Aggregate (binary: sel, seq → seq) ---

function aggregate(sel::RaspValue, vals::RaspValue)::RaspValue
    (sel.tag != :sel || vals.tag != :seq) && return sentinel_seq(_infer_n(sel, vals))
    nrows, ncols = size(sel.sel)
    vs = if length(vals.seq) == 1; fill(vals.seq[1], ncols)
    elseif length(vals.seq) == ncols; vals.seq
    else; return sentinel_seq(nrows); end
    result = Vector{Float64}(undef, nrows)
    @inbounds for i in 1:nrows
        c, s = 0, 0.0
        for j in 1:ncols; sel.sel[i, j] && (s += vs[j]; c += 1); end
        result[i] = c == 0 ? 0.0 : s / c
    end
    RaspValue(result)
end

# --- Elementwise sequence ops (binary: seq, seq → seq) ---

function seq_add(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(sa .+ sb)
end

function seq_sub(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(sa .- sb)
end

function seq_mul(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(sa .* sb)
end

function seq_eq(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(Float64.(sa .== sb))
end

# ── 3. SR interface overloads ──────────────────────────────────────────────────

import DynamicExpressions: count_scalar_constants
count_scalar_constants(::RaspValue) = 1

import SymbolicRegression: init_value
init_value(::Type{RaspValue}) = RaspValue(Float64[0.0])

import SymbolicRegression: sample_value
sample_value(rng::AbstractRNG, ::Type{RaspValue}, _) = RaspValue(Float64[randn(rng)])

import SymbolicRegression.InterfaceDynamicExpressionsModule: string_constant
function string_constant(val::RaspValue, ::Val{precision}, _) where {precision}
    val.tag == :seq || return "<sel $(size(val.sel))>"
    length(val.seq) == 1 && return string(round(val.seq[1]; digits=Int(precision)))
    return "[" * join(round.(val.seq; digits=Int(precision)), ",") * "]"
end

import SymbolicRegression.ConstantOptimizationModule: can_optimize
can_optimize(::Type{RaspValue}, _) = false

import SymbolicRegression: mutate_value
function mutate_value(rng::AbstractRNG, val::RaspValue, T, options)
    val.tag == :seq && length(val.seq) == 1 || return val
    RaspValue(Float64[val.seq[1] + randn(rng) * clamp(Float64(T), 0.01, 1.0)])
end

# ── 4. Loss ────────────────────────────────────────────────────────────────────

const RASP_BAD_LOSS = 1e9

function rasp_loss(predicted::RaspValue, target::RaspValue)::Float64
    (predicted.tag != :seq || target.tag != :seq) && return RASP_BAD_LOSS
    p, t = predicted.seq, target.seq
    length(p) != length(t) && return RASP_BAD_LOSS
    any(isnan, p) && return RASP_BAD_LOSS
    return sum((p .- t) .^ 2) / length(p)
end

# ── 5. Data generation ─────────────────────────────────────────────────────────

# ── 5a. Known RASP solutions ─────────────────────────────────────────────────
# These tasks have compact, known RASP programs built from the operators above.

# Prefix sum: cumsum(tokens)
# RASP: seq_mul(aggregate(select_leq(idx, idx), tok), seq_add(idx, [1.0]))
# Complexity ~8
function make_prefix_sum_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(cumsum(toks))
    end
    return X, y
end

# Shift left by 1: [tok[1], tok[2], ..., tok[n-1], 0]
# RASP: aggregate(select_eq(idx, seq_add(idx, [1.0])), tok)  -- default 0 for last pos
# Complexity ~6
function make_shift_left_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue([toks[2:end]; 0.0])
    end
    return X, y
end

# Consecutive duplicate detection: 1.0 where tok[i] == tok[i+1], else 0.0
#     Last position is always 0.0 (no next element).
# RASP: seq_eq(tok, shift_left(tok))  where shift_left = aggregate(select_eq(idx, idx+1), tok)
# Complexity ~8
function make_consecutive_dup_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        # Use small alphabet to get frequent duplicates
        toks = Float64.(rand(rng, 1:4, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        result = zeros(n)
        for k in 1:n-1
            result[k] = toks[k] == toks[k+1] ? 1.0 : 0.0
        end
        y[i] = RaspValue(result)
    end
    return X, y
end

# ── 5b. Challenge tasks ──────────────────────────────────────────────────────
# No known compact RASP form — these test whether SR can discover novel programs.

# Positional arithmetic: every position outputs tok[0] + tok[1] - tok[2]
# RASP (if one exists): seq_add(seq_sub(aggregate(select_eq(idx,[0]), tok),
#                                        aggregate(select_eq(idx,[2]), tok)),
#                                aggregate(select_eq(idx,[1]), tok))
# Complexity ~14
function make_positional_arith_dataset(; n_samples=128, seq_lens=4:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        val = toks[1] + toks[2] - toks[3]  # 1-indexed: positions 0,1,2
        y[i] = RaspValue(fill(val, n))
    end
    return X, y
end

# Pattern detection: 1.0 at position i if tok[i]==1 AND tok[i+1]==2, else 0.0
#     Detects the consecutive pattern [1, 2] starting at each position.
# RASP (if one exists): seq_mul(seq_eq(tok, [1.0]), seq_eq(shift_left(tok), [2.0]))
# Complexity ~12
function make_pattern_12_dataset(; n_samples=128, seq_lens=4:7, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        # Small alphabet to make patterns [1,2] likely
        toks = Float64.(rand(rng, 1:3, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        result = zeros(n)
        for k in 1:n-1
            result[k] = (toks[k] == 1.0 && toks[k+1] == 2.0) ? 1.0 : 0.0
        end
        y[i] = RaspValue(result)
    end
    return X, y
end

# ── 6. Unit tests ──────────────────────────────────────────────────────────────

@testset "New operators: seq_mul and seq_eq" begin
    @test seq_mul(RaspValue([2.0, 3.0]), RaspValue([4.0, 5.0])).seq == [8.0, 15.0]
    @test seq_mul(RaspValue([2.0, 3.0]), RaspValue([10.0])).seq == [20.0, 30.0]
    @test seq_eq(RaspValue([1.0, 2.0, 3.0]), RaspValue([1.0, 5.0, 3.0])).seq == [1.0, 0.0, 1.0]
    @test seq_eq(RaspValue([1.0, 2.0]), RaspValue([1.0])).seq == [1.0, 0.0]

    # Sentinel on type mismatch
    @test all(isnan, seq_mul(RaspValue(fill(true, 2, 2)), RaspValue([1.0])).seq)
    @test all(isnan, seq_eq(RaspValue(fill(true, 2, 2)), RaspValue([1.0])).seq)
end

@testset "Advanced composed programs" begin
    tokens = RaspValue([3.0, 1.0, 4.0, 1.0, 5.0])
    indices = RaspValue([0.0, 1.0, 2.0, 3.0, 4.0])

    # prefix sum: seq_mul(aggregate(select_leq(idx, idx), tok), seq_add(idx, [1.0]))
    cum_mean = aggregate(select_leq(indices, indices), tokens)
    prefix_sum = seq_mul(cum_mean, seq_add(indices, RaspValue([1.0])))
    @test prefix_sum.seq ≈ [3.0, 4.0, 8.0, 9.0, 14.0]

    # shift left: aggregate(select_eq(idx, seq_add(idx, [1.0])), tok)
    shifted_left = aggregate(select_eq(indices, seq_add(indices, RaspValue([1.0]))), tokens)
    @test shifted_left.seq == [1.0, 4.0, 1.0, 5.0, 0.0]

    # consecutive duplicate detection: seq_eq(tok, shift_left(tok))
    dup_tok = RaspValue([2.0, 2.0, 3.0, 3.0, 1.0])
    dup_shifted = aggregate(select_eq(indices, seq_add(indices, RaspValue([1.0]))), dup_tok)
    consec_dup = seq_eq(dup_tok, dup_shifted)
    @test consec_dup.seq == [1.0, 0.0, 1.0, 0.0, 0.0]

    # positional arithmetic: tok[0] + tok[1] - tok[2] = 3 + 1 - 4 = 0
    t0 = aggregate(select_eq(indices, RaspValue([0.0])), tokens)
    t1 = aggregate(select_eq(indices, RaspValue([1.0])), tokens)
    t2 = aggregate(select_eq(indices, RaspValue([2.0])), tokens)
    arith = seq_add(seq_sub(t0, t2), t1)
    @test arith.seq ≈ fill(0.0, 5)

    # pattern [1, 2] detection: seq_mul(seq_eq(tok, [1]), seq_eq(shift_left(tok), [2]))
    pat_tok = RaspValue([1.0, 2.0, 1.0, 2.0, 3.0])
    pat_shifted = aggregate(select_eq(indices, seq_add(indices, RaspValue([1.0]))), pat_tok)
    is_1 = seq_eq(pat_tok, RaspValue([1.0]))
    is_next_2 = seq_eq(pat_shifted, RaspValue([2.0]))
    pattern = seq_mul(is_1, is_next_2)
    @test pattern.seq == [1.0, 0.0, 1.0, 0.0, 0.0]
end

# ── 7. SR integration ─────────────────────────────────────────────────────────
# Choose a task:
#   :prefix_sum  :shift_left  :consecutive_dup  :positional_arith  :pattern_12
TASK = :prefix_sum

datasets = Dict(
    :prefix_sum      => make_prefix_sum_dataset,
    :shift_left      => make_shift_left_dataset,
    :consecutive_dup => make_consecutive_dup_dataset,
    :positional_arith => make_positional_arith_dataset,
    :pattern_12      => make_pattern_12_dataset,
)
X, y = datasets[TASK](; n_samples=128, seq_lens=3:6)

model = SRRegressor(;
    binary_operators=(
        select_eq, select_lt, select_gt, select_leq, select_true,
        aggregate, seq_add, seq_sub, seq_mul, seq_eq,
    ),
    unary_operators=(selector_width,),
    operator_enum_constructor=GenericOperatorEnum,
    elementwise_loss=rasp_loss,
    loss_type=Float64,
    maxsize=18,
    niterations=500,
    batching=true,
    batch_size=32,
    parsimony=0.01,
    adaptive_parsimony_scaling=40.0,
    warmup_maxsize_by=0.2,
    mutation_weights=MutationWeights(; mutate_constant=1.0, add_node=2.0, insert_node=2.0),
    early_stop_condition=(l, c) -> l < 1e-6,
)

mach = machine(model, X, y; scitype_check_level=0)
fit!(mach)

@testset "SR integration smoke test ($TASK)" begin
    r = report(mach)
    best_eq = r.equations[end]
    ŷ = best_eq(MLJBase.matrix(X; transpose=true))
    mean_loss = sum(rasp_loss(ŷ[i], y[i]) for i in eachindex(y)) / length(y)
    @test mean_loss < 20.0
end
