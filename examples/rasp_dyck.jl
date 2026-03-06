#! format: off

# Dyck language tasks for SymbolicRegression.jl
# Tests whether SR can discover RASP programs for balanced-parenthesis problems.
#
# Dyck-1: single bracket type  ( )     encoded as +1 / -1
# Dyck-2: two bracket types    ( ) [ ] encoded as +1 / -1 / +2 / -2
#
# Tasks (increasing difficulty):
#   :dyck1_depth         — nesting depth at each position (= prefix sum)
#   :dyck1_balance       — is the full sequence balanced? (sum == 0)
#   :dyck1_prefix_valid  — is each prefix valid? (depth >= 0 at every position so far)
#   :dyck1_valid         — full validity (balanced AND all prefixes non-negative)
#   :dyck2_depth         — total nesting depth for two bracket types
#   :dyck2_valid         — full Dyck-2 validity (no interleaving like ( [ ) ] )

using SymbolicRegression
using DynamicExpressions: GenericOperatorEnum
using MLJBase: machine, fit!, report, MLJBase
using Random
using Random: AbstractRNG
using Test

# ── 1. RaspValue type (same as rasp_advanced.jl) ─────────────────────────────

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

# --- Unary: seq_not (seq → seq) ---

function seq_not(a::RaspValue)::RaspValue
    a.tag != :seq && return sentinel_seq(_infer_n(a))
    RaspValue(Float64.(a.seq .== 0))
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

function seq_neq(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(Float64.(sa .!= sb))
end

function seq_lt(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(Float64.(sa .< sb))
end

function seq_gt(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(Float64.(sa .> sb))
end

function seq_and(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(Float64.((sa .!= 0) .& (sb .!= 0)))
end

function seq_or(a::RaspValue, b::RaspValue)::RaspValue
    (a.tag != :seq || b.tag != :seq) && return sentinel_seq(_infer_n(a, b))
    sa, sb = _broadcast_seqs(a.seq, b.seq); RaspValue(Float64.((sa .!= 0) .| (sb .!= 0)))
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

# ── 5. Helper: parenthesis display ────────────────────────────────────────────

function tokens_to_parens(toks::Vector{Float64}; dyck2=false)
    if dyck2
        map(toks) do t
            t == 1.0 ? "(" : t == -1.0 ? ")" : t == 2.0 ? "[" : t == -2.0 ? "]" : "?"
        end |> join
    else
        map(t -> t > 0 ? "(" : ")", toks) |> join
    end
end

# ── 6. Data generation ────────────────────────────────────────────────────────

# Helper: generate a valid Dyck-1 word of length n (n must be even)
function random_valid_dyck1(rng, n)
    @assert iseven(n)
    for _ in 1:10_000
        seq = shuffle(rng, [fill(1.0, n ÷ 2); fill(-1.0, n ÷ 2)])
        all(cumsum(seq) .>= 0) && return seq
    end
    # Fallback: nested structure ()()...
    return repeat([1.0, -1.0], n ÷ 2)
end

# Helper: generate a valid Dyck-2 word of length n (n must be even)
function random_valid_dyck2(rng, n)
    @assert iseven(n)
    result = Float64[]
    function _gen(remaining)
        remaining == 0 && return
        typ = rand(rng, [1.0, 2.0])
        push!(result, typ)          # open
        inner = rand(rng, 0:2:remaining-2)
        _gen(inner)
        push!(result, -typ)         # close
        _gen(remaining - 2 - inner)
    end
    _gen(n)
    return result
end

# ── 6a. Dyck-1 tasks ──────────────────────────────────────────────────────────

# Task A: Dyck-1 Depth — nesting depth at each position
# RASP: seq_mul(aggregate(select_leq(idx, idx), tok), seq_add(idx, [1.0]))  (prefix sum)
function make_dyck1_depth_dataset(; n_samples=128, seq_lens=4:2:10, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        # Mix of valid and random sequences
        toks = if rand(rng) < 0.5
            random_valid_dyck1(rng, n)
        else
            Float64.(rand(rng, [-1.0, 1.0], n))
        end
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        y[i] = RaspValue(cumsum(toks))
    end
    return X, y
end

# Task B: Dyck-1 Balance — is the full sequence balanced? (sum == 0)
# RASP: seq_eq(aggregate(select_true(tok, tok), tok), [0.0])
function make_dyck1_balance_dataset(; n_samples=128, seq_lens=4:2:10, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        # ~50/50 valid vs random for class balance
        toks = if rand(rng) < 0.5
            random_valid_dyck1(rng, n)
        else
            Float64.(rand(rng, [-1.0, 1.0], n))
        end
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        balanced = sum(toks) == 0 ? 1.0 : 0.0
        y[i] = RaspValue(fill(balanced, n))
    end
    return X, y
end

# Task C: Dyck-1 Prefix Valid — is each prefix valid? (depth >= 0 at every position so far)
# At position i: 1.0 if min(depth[0..i]) >= 0, else 0.0
function make_dyck1_prefix_valid_dataset(; n_samples=128, seq_lens=4:2:10, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = if rand(rng) < 0.5
            random_valid_dyck1(rng, n)
        else
            Float64.(rand(rng, [-1.0, 1.0], n))
        end
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        depth = cumsum(toks)
        result = zeros(n)
        for k in 1:n
            result[k] = minimum(depth[1:k]) >= 0 ? 1.0 : 0.0
        end
        y[i] = RaspValue(result)
    end
    return X, y
end

# Task D: Dyck-1 Valid — full validity (balanced AND all prefixes non-negative)
# 1.0 if valid Dyck word, else 0.0 (broadcast)
function make_dyck1_valid_dataset(; n_samples=128, seq_lens=4:2:10, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = if rand(rng) < 0.5
            random_valid_dyck1(rng, n)
        else
            Float64.(rand(rng, [-1.0, 1.0], n))
        end
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        depth = cumsum(toks)
        valid = (depth[end] == 0 && minimum(depth) >= 0) ? 1.0 : 0.0
        y[i] = RaspValue(fill(valid, n))
    end
    return X, y
end

# ── 6b. Dyck-2 tasks ──────────────────────────────────────────────────────────
# Encoding: ( → +1, ) → -1, [ → +2, ] → -2

# Task E: Dyck-2 Depth — total nesting depth (counting both bracket types)
# depth[i] = cumsum(sign.(tokens))[i]
function make_dyck2_depth_dataset(; n_samples=128, seq_lens=4:2:10, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = if rand(rng) < 0.5
            random_valid_dyck2(rng, n)
        else
            Float64.(rand(rng, [-1.0, 1.0, -2.0, 2.0], n))
        end
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        signed = Float64.(sign.(toks))  # +1 for open, -1 for close
        y[i] = RaspValue(cumsum(signed))
    end
    return X, y
end

# Task F: Dyck-2 Valid — full Dyck-2 validity (properly nested, no interleaving)
# A sequence is valid iff it can be generated by the Dyck-2 grammar.
# Requires stack-like tracking — a genuine challenge for RASP/transformers.
function _is_valid_dyck2(toks::Vector{Float64})
    stack = Float64[]
    for t in toks
        if t > 0  # open bracket
            push!(stack, t)
        else  # close bracket
            isempty(stack) && return false
            pop!(stack) != -t && return false  # type mismatch
        end
    end
    return isempty(stack)
end

function make_dyck2_valid_dataset(; n_samples=128, seq_lens=4:2:10, rng=Random.MersenneTwister(0))
    X = Vector{@NamedTuple{tokens::RaspValue, indices::RaspValue}}(undef, n_samples)
    y = Vector{RaspValue}(undef, n_samples)
    for i in 1:n_samples
        n = rand(rng, seq_lens)
        toks = if rand(rng) < 0.5
            random_valid_dyck2(rng, n)
        else
            Float64.(rand(rng, [-1.0, 1.0, -2.0, 2.0], n))
        end
        X[i] = (; tokens=RaspValue(toks), indices=RaspValue(Float64.(0:n-1)))
        valid = _is_valid_dyck2(toks) ? 1.0 : 0.0
        y[i] = RaspValue(fill(valid, n))
    end
    return X, y
end

# ── 7. Unit tests ──────────────────────────────────────────────────────────────

@testset "Dyck-1 RASP programs" begin
    # Depth via prefix sum
    toks = RaspValue([1.0, 1.0, -1.0, 1.0, -1.0, -1.0])
    idx  = RaspValue([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cum_mean = aggregate(select_leq(idx, idx), toks)
    depth = seq_mul(cum_mean, seq_add(idx, RaspValue([1.0])))
    @test depth.seq ≈ [1.0, 2.0, 1.0, 2.0, 1.0, 0.0]

    # Balance check: seq_eq(aggregate(select_true(tok, tok), tok), [0.0])
    balanced_toks = RaspValue([1.0, -1.0, 1.0, -1.0])
    mean_val = aggregate(select_true(balanced_toks, balanced_toks), balanced_toks)
    is_balanced = seq_eq(mean_val, RaspValue([0.0]))
    @test is_balanced.seq == [1.0, 1.0, 1.0, 1.0]

    unbalanced_toks = RaspValue([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
    mean_val2 = aggregate(select_true(unbalanced_toks, unbalanced_toks), unbalanced_toks)
    is_balanced2 = seq_eq(mean_val2, RaspValue([0.0]))
    @test is_balanced2.seq == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # sum is 0 so balanced
end

@testset "Dyck-2 validity" begin
    @test _is_valid_dyck2([1.0, -1.0, 2.0, -2.0]) == true    # ()[]
    @test _is_valid_dyck2([1.0, 2.0, -2.0, -1.0]) == true     # ([])
    @test _is_valid_dyck2([1.0, 2.0, -1.0, -2.0]) == false    # ([)] interleaved
    @test _is_valid_dyck2([1.0, 1.0, -1.0]) == false           # unbalanced
    @test _is_valid_dyck2([2.0, -1.0]) == false                 # type mismatch
    @test _is_valid_dyck2(Float64[]) == true                    # empty is valid
end

@testset "Dyck dataset generation" begin
    X1, y1 = make_dyck1_depth_dataset(; n_samples=10)
    @test length(X1) == 10
    @test all(x -> x.tokens.tag == :seq, X1)

    X2, y2 = make_dyck1_balance_dataset(; n_samples=10)
    @test all(y -> all(v -> v == 0.0 || v == 1.0, y.seq), y2)

    X3, y3 = make_dyck1_valid_dataset(; n_samples=10)
    @test all(y -> all(v -> v == 0.0 || v == 1.0, y.seq), y3)

    X4, y4 = make_dyck2_depth_dataset(; n_samples=10)
    @test length(X4) == 10

    X5, y5 = make_dyck2_valid_dataset(; n_samples=10)
    @test all(y -> all(v -> v == 0.0 || v == 1.0, y.seq), y5)
end

# ── 8. SR integration ─────────────────────────────────────────────────────────

TASK = :dyck2_valid

TASK_INFO = Dict(
    :dyck1_depth        => (name="Dyck-1 Depth",         desc="Nesting depth at each position (= prefix sum of signed tokens)", encoding="( → +1, ) → -1", dyck2=false),
    :dyck1_balance      => (name="Dyck-1 Balance",       desc="Is the full sequence balanced? (sum == 0)", encoding="( → +1, ) → -1", dyck2=false),
    :dyck1_prefix_valid => (name="Dyck-1 Prefix Valid",  desc="Is each prefix valid? (depth >= 0 at every position so far)", encoding="( → +1, ) → -1", dyck2=false),
    :dyck1_valid        => (name="Dyck-1 Validity",      desc="Full validity: balanced AND all prefixes non-negative", encoding="( → +1, ) → -1", dyck2=false),
    :dyck2_depth        => (name="Dyck-2 Depth",         desc="Total nesting depth for two bracket types", encoding="( → +1, ) → -1, [ → +2, ] → -2", dyck2=true),
    :dyck2_valid        => (name="Dyck-2 Validity",      desc="Full Dyck-2 validity (no interleaving like ( [ ) ] )", encoding="( → +1, ) → -1, [ → +2, ] → -2", dyck2=true),
)

datasets = Dict(
    :dyck1_depth        => make_dyck1_depth_dataset,
    :dyck1_balance      => make_dyck1_balance_dataset,
    :dyck1_prefix_valid => make_dyck1_prefix_valid_dataset,
    :dyck1_valid        => make_dyck1_valid_dataset,
    :dyck2_depth        => make_dyck2_depth_dataset,
    :dyck2_valid        => make_dyck2_valid_dataset,
)

info = TASK_INFO[TASK]
println("╔", "═"^68, "╗")
println("║ Task: ", info.name, " "^max(0, 60 - length(info.name)), "║")
println("║ ", info.desc, " "^max(0, 66 - length(info.desc)), "║")
println("║ Encoding: ", info.encoding, " "^max(0, 56 - length(info.encoding)), "║")
println("╚", "═"^68, "╝")

X, y = datasets[TASK](; n_samples=128, seq_lens=4:2:10)

# Show a few training examples
println("\n── Training examples ──")
for i in 1:min(3, length(X))
    toks = X[i].tokens.seq
    println("  $(tokens_to_parens(toks; dyck2=info.dyck2))  →  target = $(y[i].seq)")
end

model = SRRegressor(;
    binary_operators=(
        select_eq, select_lt, select_gt, select_leq, select_true,
        aggregate, seq_add, seq_sub, seq_mul, seq_eq, seq_neq, seq_lt, seq_gt, seq_and, seq_or,
    ),
    unary_operators=(selector_width, seq_not),
    operator_enum_constructor=GenericOperatorEnum,
    elementwise_loss=rasp_loss,
    loss_type=Float64,
    maxsize=25,
    niterations=1000,
    batching=true,
    batch_size=32,
    parsimony=0.02,
    adaptive_parsimony_scaling=40.0,
    warmup_maxsize_by=0.2,
    mutation_weights=MutationWeights(; mutate_constant=1.0, add_node=2.0, insert_node=2.0),
    early_stop_condition=(l, c) -> l < 1e-6,
)

println("\n── Starting SR search (niterations=50) ──")
mach = machine(model, X, y; scitype_check_level=0)
fit!(mach)

# ── 9. Results ─────────────────────────────────────────────────────────────────

let r = report(mach)
    best_eq = r.equations[end]
    println()
    println("╔", "═"^68, "╗")
    println("║ Results: ", info.name, " "^max(0, 57 - length(info.name)), "║")
    println("╚", "═"^68, "╝")
    println("Best equation (complexity $(r.complexities[end])):")
    println("  ", best_eq)

    n_show = min(5, length(X))
    Xmat = MLJBase.matrix(X; transpose=true)
    println()
    for i in 1:n_show
        pred = best_eq(Xmat[:, i:i])[1]
        toks = X[i].tokens.seq
        loss = rasp_loss(pred, y[i])
        println("Sample $i:  $(tokens_to_parens(toks; dyck2=info.dyck2))")
        println("  tokens  = ", toks)
        println("  indices = ", X[i].indices.seq)
        println("  target  = ", y[i].seq)
        println("  predict = ", pred.tag == :seq ? pred.seq : pred)
        println("  loss    = ", loss)
        println()
    end

    # Summary
    ŷ = best_eq(Xmat)
    mean_loss = sum(rasp_loss(ŷ[i], y[i]) for i in eachindex(y)) / length(y)
    perfect = count(i -> rasp_loss(ŷ[i], y[i]) < 1e-6, eachindex(y))
    println("── Summary ──")
    println("  Mean loss:      ", round(mean_loss; digits=6))
    println("  Perfect:        $perfect / $(length(y))")
    println("  Complexity:     ", r.complexities[end])
end
