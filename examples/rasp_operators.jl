#! format: off

using SymbolicRegression
using DynamicExpressions: GenericOperatorEnum
using MLJBase: machine, fit!, report, MLJBase
using Random
using Random: AbstractRNG
using Test

# ── 1. RaspValue type ──────────────────────────────────────────────────────────

"""
Tagged union holding either a sequence (SOp) or a selector (boolean matrix).
SR expression trees require a single type; RaspValue wraps both RASP value kinds.
"""
struct RaspValue
    tag::Symbol            # :seq or :sel
    seq::Vector{Float64}   # populated when tag == :seq
    sel::Matrix{Bool}      # populated when tag == :sel
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
function _infer_n(a::RaspValue)
    a.tag == :seq && return length(a.seq)
    return size(a.sel, 1)
end

"""Broadcast length-1 (scalar constant) sequences to match length n."""
function _broadcast_seqs(a::Vector{Float64}, b::Vector{Float64})
    la, lb = length(a), length(b)
    la == lb && return (a, b)
    la == 1 && return (fill(a[1], lb), b)
    lb == 1 && return (a, fill(b[1], la))
    n = max(la, lb)
    return (fill(NaN, n), fill(NaN, n))
end

function Base.:(==)(a::RaspValue, b::RaspValue)
    a.tag != b.tag && return false
    a.tag == :seq && return a.seq == b.seq
    return a.sel == b.sel
end
function Base.hash(a::RaspValue, h::UInt)
    h = hash(a.tag, h)
    a.tag == :seq ? hash(a.seq, h) : hash(a.sel, h)
end
function Base.isnan(a::RaspValue)
    a.tag == :seq && return any(isnan, a.seq)
    return false
end
function Base.copy(a::RaspValue)
    RaspValue(a.tag, copy(a.seq), copy(a.sel))
end

# ── 2. Operators ────────────────────────────────────────────────────────────────

# Binary: Select operators (seq, seq → sel)

function select_eq(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    sel = Matrix{Bool}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        sel[i, j] = ks[j] == qs[i]
    end
    RaspValue(sel)
end

function select_lt(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    sel = Matrix{Bool}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        sel[i, j] = ks[j] < qs[i]
    end
    RaspValue(sel)
end

function select_leq(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    sel = Matrix{Bool}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        sel[i, j] = ks[j] <= qs[i]
    end
    RaspValue(sel)
end

function select_true(keys::RaspValue, queries::RaspValue)::RaspValue
    (keys.tag != :seq || queries.tag != :seq) && return sentinel_sel(_infer_n(keys, queries))
    ks, qs = _broadcast_seqs(keys.seq, queries.seq)
    n = length(ks)
    RaspValue(fill(true, n, n))
end

# Unary: SelectorWidth (sel → seq)

function selector_width(x::RaspValue)::RaspValue
    x.tag != :sel && return sentinel_seq(_infer_n(x))
    n = size(x.sel, 1)
    result = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        result[i] = Float64(count(x.sel[i, :]))
    end
    RaspValue(result)
end

# Binary: Aggregate (sel, seq → seq)

function aggregate(sel::RaspValue, vals::RaspValue)::RaspValue
    (sel.tag != :sel || vals.tag != :seq) && return sentinel_seq(_infer_n(sel, vals))
    nrows = size(sel.sel, 1)
    ncols = size(sel.sel, 2)
    vs = if length(vals.seq) == 1
        fill(vals.seq[1], ncols)
    elseif length(vals.seq) == ncols
        vals.seq
    else
        return sentinel_seq(nrows)
    end
    result = Vector{Float64}(undef, nrows)
    @inbounds for i in 1:nrows
        c = 0
        s = 0.0
        for j in 1:ncols
            if sel.sel[i, j]
                s += vs[j]
                c += 1
            end
        end
        result[i] = c == 0 ? 0.0 : s / c
    end
    RaspValue(result)
end

# Binary: Elementwise sequence arithmetic (seq, seq → seq)

function seq_add(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq)
    RaspValue(sa .+ sb)
end

function seq_sub(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq)
    RaspValue(sa .- sb)
end

# ── 3. SR interface overloads ───────────────────────────────────────────────────

import DynamicExpressions: count_scalar_constants
count_scalar_constants(::RaspValue) = 1

import SymbolicRegression: init_value
init_value(::Type{RaspValue}) = RaspValue(Float64[0.0])

import SymbolicRegression: sample_value
sample_value(rng::AbstractRNG, ::Type{RaspValue}, _) = RaspValue(Float64[randn(rng)])

import SymbolicRegression.InterfaceDynamicExpressionsModule: string_constant
function string_constant(val::RaspValue, ::Val{precision}, _) where {precision}
    if val.tag == :seq
        if length(val.seq) == 1
            return string(round(val.seq[1]; digits=Int(precision)))
        else
            return "[" * join(round.(val.seq; digits=Int(precision)), ",") * "]"
        end
    else
        return "<sel $(size(val.sel))>"
    end
end

import SymbolicRegression.ConstantOptimizationModule: can_optimize
can_optimize(::Type{RaspValue}, _) = false

import SymbolicRegression: mutate_value
function mutate_value(rng::AbstractRNG, val::RaspValue, T, options)
    if val.tag == :seq && length(val.seq) == 1
        noise = randn(rng) * clamp(Float64(T), 0.01, 1.0)
        return RaspValue(Float64[val.seq[1] + noise])
    end
    return val
end

# ── 4. Loss function ───────────────────────────────────────────────────────────

const RASP_BAD_LOSS = 1e9

function rasp_loss(predicted::RaspValue, target::RaspValue)::Float64
    (predicted.tag != :seq || target.tag != :seq) && return RASP_BAD_LOSS
    p, t = predicted.seq, target.seq
    length(p) != length(t) && return RASP_BAD_LOSS
    any(isnan, p) && return RASP_BAD_LOSS
    return sum((p .- t) .^ 2) / length(p)
end

# ── 5. Data generation ─────────────────────────────────────────────────────────

function make_hist_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue([Float64(count(==(toks[k]), toks)) for k in 1:n])
    end
    return X, y
end

function make_reverse_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(reverse(toks))
    end
    return X, y
end

function make_sort_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(randperm(rng, 20)[1:n])
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(sort(toks))
    end
    return X, y
end

function make_minimum_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(randperm(rng, 20)[1:n])
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(fill(minimum(toks), n))
    end
    return X, y
end

function make_maximum_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(randperm(rng, 20)[1:n])
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(fill(maximum(toks), n))
    end
    return X, y
end

function make_sort_descending_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(randperm(rng, 20)[1:n])
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(sort(toks; rev=true))
    end
    return X, y
end

function make_shift_right_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue([0.0; toks[1:end-1]])  # shift right, 0 fills position 0
    end
    return X, y
end

function make_mean_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(fill(sum(toks) / n, n))
    end
    return X, y
end

function make_cumulative_mean_dataset(; n_samples=128, seq_lens=3:6, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = Float64.(rand(rng, 1:10, n))
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue([sum(toks[1:k]) / k for k in 1:n])
    end
    return X, y
end

# ── 6. Unit tests ──────────────────────────────────────────────────────────────

@testset "RASP operator correctness" begin
    # select_eq
    r = select_eq(RaspValue([1.0, 2.0, 1.0]), RaspValue([1.0, 2.0, 1.0]))
    @test r.tag == :sel
    @test r.sel == Bool[1 0 1; 0 1 0; 1 0 1]

    # select_lt
    r = select_lt(RaspValue([1.0, 2.0, 3.0]), RaspValue([2.0, 2.0, 2.0]))
    @test r.sel == Bool[1 0 0; 1 0 0; 1 0 0]

    # select_true
    r = select_true(RaspValue([1.0, 2.0]), RaspValue([3.0, 4.0]))
    @test r.sel == fill(true, 2, 2)

    # selector_width
    r = selector_width(RaspValue(Bool[1 1 0; 0 0 1; 1 1 1]))
    @test r.tag == :seq
    @test r.seq == [2.0, 1.0, 3.0]

    # aggregate
    r = aggregate(RaspValue(Bool[1 1 0; 0 0 1]), RaspValue([10.0, 20.0, 30.0]))
    @test r.seq ≈ [15.0, 30.0]

    # aggregate with no selection → default 0
    r = aggregate(RaspValue(Bool[0 0; 1 0]), RaspValue([5.0, 10.0]))
    @test r.seq[1] == 0.0
    @test r.seq[2] == 5.0

    # seq_add / seq_sub
    @test seq_add(RaspValue([1.0, 2.0]), RaspValue([3.0, 4.0])).seq == [4.0, 6.0]
    @test seq_sub(RaspValue([5.0, 3.0]), RaspValue([1.0, 1.0])).seq == [4.0, 2.0]

    # scalar broadcast
    @test seq_add(RaspValue([1.0, 2.0]), RaspValue([10.0])).seq == [11.0, 12.0]
    @test seq_sub(RaspValue([10.0]), RaspValue([1.0, 2.0])).seq == [9.0, 8.0]
end

@testset "RASP sentinel / no-throw behavior" begin
    # Type mismatch: selector_width on seq → sentinel
    r = selector_width(RaspValue([1.0, 2.0]))
    @test r.tag == :seq
    @test all(isnan, r.seq)

    # Type mismatch: select_eq on sel → sentinel
    r = select_eq(RaspValue(fill(true, 2, 2)), RaspValue([1.0, 2.0]))
    @test r.tag == :sel

    # Type mismatch: aggregate wrong arg order → sentinel
    r = aggregate(RaspValue([1.0, 2.0]), RaspValue(fill(true, 2, 2)))
    @test r.tag == :seq
    @test all(isnan, r.seq)

    # Type mismatch: seq_add with sel → sentinel
    r = seq_add(RaspValue(fill(true, 2, 2)), RaspValue([1.0, 2.0]))
    @test all(isnan, r.seq)
end

@testset "RASP composed programs" begin
    tokens = RaspValue([3.0, 1.0, 3.0, 2.0, 3.0])
    indices = RaspValue([0.0, 1.0, 2.0, 3.0, 4.0])

    # hist: selector_width(select_eq(tokens, tokens))
    hist = selector_width(select_eq(tokens, tokens))
    @test hist.seq == [3.0, 1.0, 3.0, 1.0, 3.0]

    # length: selector_width(select_true(tokens, tokens))
    len = selector_width(select_true(tokens, tokens))
    @test len.seq == [5.0, 5.0, 5.0, 5.0, 5.0]

    # reverse: aggregate(select_eq(indices, len - indices - 1), tokens)
    rev_idx = seq_sub(seq_sub(len, indices), RaspValue([1.0]))
    @test rev_idx.seq == [4.0, 3.0, 2.0, 1.0, 0.0]
    rev_sel = select_eq(indices, rev_idx)
    reversed = aggregate(rev_sel, tokens)
    @test reversed.seq == [3.0, 2.0, 3.0, 1.0, 3.0]

    # sort (unique elements): aggregate(select_eq(rank, indices), tokens)
    #   where rank = selector_width(select_lt(tokens, tokens))
    unique_tokens = RaspValue([5.0, 2.0, 8.0, 1.0])
    unique_indices = RaspValue([0.0, 1.0, 2.0, 3.0])
    rank = selector_width(select_lt(unique_tokens, unique_tokens))
    @test rank.seq == [2.0, 1.0, 3.0, 0.0]
    sorted = aggregate(select_eq(rank, unique_indices), unique_tokens)
    @test sorted.seq == [1.0, 2.0, 5.0, 8.0]

    # minimum: aggregate(select_eq(rank, [0.0]), tokens)  -- rank 0 = nothing smaller
    min_val = aggregate(select_eq(rank, RaspValue([0.0])), unique_tokens)
    @test min_val.seq == [1.0, 1.0, 1.0, 1.0]

    # maximum: aggregate(select_eq(rank, len-1), tokens)  -- rank n-1 = nothing larger
    len_u = selector_width(select_true(unique_tokens, unique_tokens))
    max_val = aggregate(select_eq(rank, seq_sub(len_u, RaspValue([1.0]))), unique_tokens)
    @test max_val.seq == [8.0, 8.0, 8.0, 8.0]

    # sort descending: aggregate(select_eq(len-1-rank, indices), tokens)
    desc_rank = seq_sub(seq_sub(len_u, rank), RaspValue([1.0]))
    @test desc_rank.seq == [1.0, 2.0, 0.0, 3.0]  # 5→1, 2→2, 8→0, 1→3
    sorted_desc = aggregate(select_eq(desc_rank, unique_indices), unique_tokens)
    @test sorted_desc.seq == [8.0, 5.0, 2.0, 1.0]

    # shift right: aggregate(select_eq(indices, indices - 1), tokens)  -- default 0
    shifted = aggregate(select_eq(indices, seq_sub(indices, RaspValue([1.0]))), tokens)
    @test shifted.seq == [0.0, 3.0, 1.0, 3.0, 2.0]

    # mean: aggregate(select_true(tokens, tokens), tokens)
    mean_val = aggregate(select_true(tokens, tokens), tokens)
    @test mean_val.seq ≈ fill(2.4, 5)  # (3+1+3+2+3)/5

    # cumulative mean: aggregate(select_leq(indices, indices), tokens)
    cm_tokens = RaspValue([4.0, 2.0, 6.0])
    cm_indices = RaspValue([0.0, 1.0, 2.0])
    cm = aggregate(select_leq(cm_indices, cm_indices), cm_tokens)
    @test cm.seq ≈ [4.0, 3.0, 4.0]  # [4/1, 6/2, 12/3]
end

@testset "RASP loss function" begin
    @test rasp_loss(RaspValue([1.0, 2.0]), RaspValue([1.0, 2.0])) == 0.0
    @test rasp_loss(RaspValue([1.0, 2.0]), RaspValue([3.0, 4.0])) == 4.0
    @test rasp_loss(RaspValue([NaN, 1.0]), RaspValue([1.0, 2.0])) == RASP_BAD_LOSS
    @test rasp_loss(RaspValue(fill(true, 2, 2)), RaspValue([1.0, 2.0])) == RASP_BAD_LOSS
    @test rasp_loss(RaspValue([1.0]), RaspValue([1.0, 2.0])) == RASP_BAD_LOSS
end

# ── 7. SR integration ──────────────────────────────────────────────────────────
# Choose a task:
#   :hist :reverse :sort :sort_descending :minimum :maximum
#   :shift_right :mean :cumulative_mean
TASK = :reverse

datasets = Dict(
    :hist            => make_hist_dataset,
    :reverse         => make_reverse_dataset,
    :sort            => make_sort_dataset,
    :sort_descending => make_sort_descending_dataset,
    :minimum         => make_minimum_dataset,
    :maximum         => make_maximum_dataset,
    :shift_right     => make_shift_right_dataset,
    :mean            => make_mean_dataset,
    :cumulative_mean => make_cumulative_mean_dataset,
)
X, y = datasets[TASK](; n_samples=128, seq_lens=3:6)

model = SRRegressor(;
    binary_operators=(select_eq, select_lt, select_leq, select_true, aggregate, seq_add, seq_sub),
    unary_operators=(selector_width,),
    operator_enum_constructor=GenericOperatorEnum,
    elementwise_loss=rasp_loss,
    loss_type=Float64,
    maxsize=20,
    niterations=2000,
    batching=true,
    batch_size=32,
    parsimony=0.01,
    adaptive_parsimony_scaling=20.0,
    early_stop_condition=(l, c) -> l < 1e-6,
    population_size=100,
)

mach = machine(model, X, y; scitype_check_level=0)
fit!(mach)

@testset "SR integration smoke test ($TASK)" begin
    r = report(mach)
    best_eq = r.equations[end]
    ŷ = best_eq(MLJBase.matrix(X; transpose=true))
    mean_loss = sum(rasp_loss(ŷ[i], y[i]) for i in eachindex(y)) / length(y)
    @test mean_loss < 10.0
end
