"""Query-cost models for naive nesting and Schrottenloher--Stevens early aborts.

SS optimization requires SciPy; evaluation and naive optimization use only the
Python standard library. No quantum circuit is simulated.
All inputs describe exact cardinalities; costs are positive costs per coherent
Boolean query, including its controlled/inverse implementation. A phase query
uses one Boolean query by kickback, a phase-adjusted predicate marker uses two,
and operations on already computed success flags cost no predicate queries.
Other gates, workspace, and the acquisition of cardinalities are not charged.

``SearchProblem.from_blocks`` accepts exactly the region ordering used by the
existing RGS scripts: region 0 satisfies every constraint, and region 2**k-1
satisfies none. Oracle 0 corresponds to the most significant region-index bit.
Alternatively, ``intersection_sizes[mask]`` is the intersection size for the
oracle indices whose bits are set in mask; entry 0 is N. These are DIFFERENT
array conventions. Query-count outputs always use the ORIGINAL oracle order;
iteration counts and stage probabilities use the chosen NESTING order.

SS implements Appendix A, Algorithms 12--13 / Lemmas 8--9, of:
  A. Schrottenloher and M. Stevens (2024),
  Quantum Procedures for Nested Search Problems: with Applications in
  Cryptanalysis, https://doi.org/10.62056/aee0fhbmo .
For ordered prefix sizes s[0]=N,...,s[k]=r, the exact recurrence is
  P[0]=1; P[i+1]=sin((2*t[i]+1)*asin(sqrt(s[i+1]/s[i]*P[i])))**2,
and the filter-query counts of one preparation are
  Q[i]=product(2*t[j]+1 for j >= i).
These counts ALREADY include uncomputation. Do not multiply them by two again.

Low-level evaluate_ss completion modes:
  amplify : coherent final amplification to target_success (its default);
  repeat  : a fixed number of measured runs reaching target_success;
  restart : repeat until success; expected cost C/P, as in Appendix A.5;
  none    : return the inner preparation without boosting its success.
The final amplification uses the stored success flag, so phase adjustment
there does not require extra predicate queries. Naive nesting fully amplifies
each intermediate intersection; its final stage may stop at target_success.
Both models count finite integer rounds, rather than leading asymptotic terms.

``optimize_naive`` is globally optimal over all orders in this naive model.
``optimize_ss`` fixes one oracle per level and uses optimize_naive's order.
``optimize_ss_for_order`` performs only the SS optimization for a supplied
order, allowing the common order search to be timed separately and reused.
It minimizes the cost AFTER final coherent amplification to target_success,
using exactly evaluate_ss(..., completion="amplify") for integer candidates.
This adapts Appendix A.5's optimization to an amplified completion, retaining
Lemma 11's per-level bounds but replacing its C/P restart objective. Exact
cardinalities set both probability bounds equal to the conditional probability.
A continuous relaxation supplies candidates; integer round counts and the
final amplification are then optimized together. Small bounded integer spaces
are enumerated; otherwise the search is local and has no global guarantee.
It does not search other orders or merge tests. Both continuous and completed
integer costs are reported. All costs here count predicate queries, rather
than the paper's full gate cost.

Example:
    problem = SearchProblem.from_blocks(
        [1, 12, 8, 79, 10, 177, 31, 2**40 - 318],
        costs=[1, 10, 10],
    )
    naive = optimize_naive(problem, target_success=0.99)
    ss = optimize_ss(problem, target_success=0.99)
    print(naive.query_counts, naive.weighted_cost)
    print(ss.query_counts, ss.weighted_cost, ss.optimality)

Run ``python nested_amplification.py`` for a small demonstration. Use
``--help`` for its options; importing the module does not run experiments.
"""

from dataclasses import asdict, dataclass, replace
import argparse
import itertools
import json
import math
import operator
from typing import Optional, Sequence, Tuple, Union


Count = Union[int, float]


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be an integer, not a Boolean")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(name + " must be an integer") from exc


def _target(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("target_success must lie in (0, 1]")
    return value


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must lie in [0, 1]")
    return value


def _grover_probability(probability, rounds):
    if probability in (0.0, 1.0):
        return probability
    angle = (2 * rounds + 1) * math.asin(math.sqrt(probability))
    return math.sin(angle) ** 2


def _amplification_plan(probability, target):
    """Minimum first-peak round count, with an adjusted last round if needed."""
    if probability >= target:
        return 0, False, probability
    if probability <= 0:
        raise ValueError("a zero-success preparation cannot be amplified")
    theta = math.asin(math.sqrt(probability))
    real_rounds = (math.asin(math.sqrt(target)) / theta - 1) / 2
    nearest = round(real_rounds)
    tolerance = 16 * math.ulp(max(1.0, abs(real_rounds)))
    if abs(real_rounds - nearest) <= tolerance:
        real_rounds = float(nearest)
    rounds = max(0, math.ceil(real_rounds))
    achieved = _grover_probability(probability, rounds)
    adjusted = achieved < target - 32 * math.ulp(target)
    return rounds, adjusted, target if adjusted else max(target, achieved)


def _weighted_cost(counts, costs):
    try:
        return math.fsum(q * c for q, c in zip(counts, costs))
    except OverflowError:
        return math.inf


@dataclass(frozen=True)
class SearchProblem:
    """Complete intersection-size data, indexed by subsets of oracle indices.

    Example for two sets: SearchProblem([N, len(A0), len(A1), r], [c0,c1]).
    Counts must be nonnegative integers, with r>0. A Mobius inversion checks
    that they describe an actual collection of sets, not only a monotone table.
    """

    intersection_sizes: Sequence[int]
    costs: Optional[Sequence[float]] = None

    def __post_init__(self):
        sizes = tuple(_integer(x, "intersection size") for x in self.intersection_sizes)
        length = len(sizes)
        if length < 2 or length & (length - 1):
            raise ValueError("intersection_sizes must have length 2**k, k>=1")
        k = length.bit_length() - 1
        if sizes[0] <= 0 or sizes[-1] <= 0 or any(x < 0 for x in sizes):
            raise ValueError("need N>0, r>0, and nonnegative intersection sizes")
        atoms = list(sizes)
        for i in range(k):
            for mask in range(length):
                if not mask & (1 << i):
                    atoms[mask] -= atoms[mask | (1 << i)]
        if any(x < 0 for x in atoms):
            raise ValueError("intersection sizes are inconsistent (a region has negative size)")
        costs = (1.0,) * k if self.costs is None else tuple(float(c) for c in self.costs)
        if len(costs) != k or any(not math.isfinite(c) or c <= 0 for c in costs):
            raise ValueError("provide one finite positive cost per oracle")
        object.__setattr__(self, "intersection_sizes", sizes)
        object.__setattr__(self, "costs", costs)

    @property
    def k(self):
        return len(self.costs)

    @property
    def N(self):
        return self.intersection_sizes[0]

    @property
    def r(self):
        return self.intersection_sizes[-1]

    @classmethod
    def from_blocks(cls, block_sizes, costs=None):
        """Convert the region ordering of grover_3_random_oracle.py to subsets."""
        blocks = tuple(_integer(x, "block size") for x in block_sizes)
        length = len(blocks)
        if length < 2 or length & (length - 1) or any(x < 0 for x in blocks):
            raise ValueError("block_sizes must be 2**k nonnegative integers, k>=1")
        k = length.bit_length() - 1
        sizes = [0] * length
        for region, size in enumerate(blocks):
            mask = sum(1 << i for i in range(k) if not region & (1 << (k - 1 - i)))
            sizes[mask] = size
        for i in range(k):
            for mask in range(length):
                if not mask & (1 << i):
                    sizes[mask] += sizes[mask | (1 << i)]
        return cls(sizes, costs)

    def prefix_sizes(self, order=None):
        order = _order(self, order)
        mask = 0
        sizes = [self.N]
        for i in order:
            mask |= 1 << i
            sizes.append(self.intersection_sizes[mask])
        return tuple(sizes)


def _order(problem, order):
    order = tuple(range(problem.k)) if order is None else tuple(_integer(i, "oracle index") for i in order)
    if sorted(order) != list(range(problem.k)):
        raise ValueError("order must be a permutation of 0,...,k-1")
    return order


@dataclass(frozen=True)
class FixedOrderProblem:
    """Known prefix sizes (N,s_0,...,s_{k-1}=r) for order (0,...,k-1).

    Needs only k+1 sizes, rather than the complete 2**k subset table.
    Use evaluate_naive, evaluate_ss, or optimize_ss_for_order; order-search
    functions require SearchProblem's complete intersection table instead.
    Any positive nonincreasing chain describes realizable intersections.
    """

    ordered_sizes: Sequence[int]
    costs: Optional[Sequence[float]] = None

    def __post_init__(self):
        sizes = tuple(_integer(x, "prefix size") for x in self.ordered_sizes)
        if len(sizes) < 2 or sizes[-1] <= 0 or any(a < b for a, b in zip(sizes, sizes[1:])):
            raise ValueError("need k+1 positive nonincreasing prefix sizes, k>=1")
        k = len(sizes) - 1
        costs = (1.0,) * k if self.costs is None else tuple(float(c) for c in self.costs)
        if len(costs) != k or any(not math.isfinite(c) or c <= 0 for c in costs):
            raise ValueError("provide one finite positive cost per oracle")
        object.__setattr__(self, "ordered_sizes", sizes)
        object.__setattr__(self, "costs", costs)

    @property
    def k(self):
        return len(self.costs)

    @property
    def N(self):
        return self.ordered_sizes[0]

    @property
    def r(self):
        return self.ordered_sizes[-1]

    def prefix_sizes(self, order=None):
        if _order(self, order) != tuple(range(self.k)):
            raise ValueError("FixedOrderProblem supports only order (0,...,k-1)")
        return self.ordered_sizes


@dataclass(frozen=True)
class NestedResult:
    """Evaluation plus search metadata. Cost is expected when completion='restart'."""

    method: str
    order: Tuple[int, ...]
    prefix_sizes: Tuple[int, ...]
    iterations: Tuple[int, ...]
    stage_success_probabilities: Tuple[float, ...]
    inner_success_probability: float
    success_probability: float
    target_success: float
    inner_query_counts: Tuple[int, ...]
    query_counts: Tuple[Count, ...]
    inner_weighted_cost: float
    weighted_cost: float
    completion: str
    phase_adjustments: Tuple[bool, ...]
    outer_iterations: int = 0
    outer_phase_adjusted: bool = False
    repetitions: Count = 1
    success_flags: int = 0
    optimality: str = "not_optimized"
    evaluations: int = 1
    orders_searched: int = 1
    orders_exhaustive: bool = False
    iteration_bounds: Tuple[int, ...] = ()
    optimization_method: str = "not_optimized"
    optimizer_success: Optional[bool] = None
    optimizer_message: str = ""
    optimizer_iterations: int = 0
    continuous_iterations: Tuple[float, ...] = ()
    continuous_success_probability: Optional[float] = None
    continuous_amplified_cost: Optional[float] = None
    # Retained for compatibility with older restart-optimization output.
    continuous_expected_cost: Optional[float] = None
    rounding_cost_ratio: Optional[float] = None
    naive_target_success: Optional[float] = None

    @property
    def total_queries(self):
        return sum(self.query_counts)

    def as_dict(self):
        """Return JSON/CSV-friendly metadata (integer counts stay Python ints)."""
        return asdict(self)


def evaluate_naive(problem, order=None, target_success=1.0):
    """Evaluate exact stagewise nesting; only the final target may be below 1.

    At a level with R rounds, the child preparation is called 2*R+1 times.
    The new marker costs R queries, plus one if its final step is phase-adjusted.
    A redundant constraint can require zero rounds and zero marker queries.
    """
    target = _target(target_success)
    order = _order(problem, order)
    sizes = problem.prefix_sizes(order)
    queries = [0] * problem.k
    rounds, adjusted, successes = [], [], []
    for level, oracle in enumerate(order):
        goal = target if level == problem.k - 1 else 1.0
        count, phase, success = _amplification_plan(sizes[level + 1] / sizes[level], goal)
        rounds.append(count)
        adjusted.append(phase)
        successes.append(success)
    # A suffix product computes all counts in O(k), without repeatedly
    # scaling a length-k vector at each level. Includes uncomputation.
    suffix = 1
    for level in range(problem.k - 1, -1, -1):
        queries[order[level]] = (rounds[level] + int(adjusted[level])) * suffix
        suffix *= 2 * rounds[level] + 1
    cost = _weighted_cost(queries, problem.costs)
    return NestedResult(
        "naive", order, sizes, tuple(rounds), tuple(successes), successes[-1],
        successes[-1], target, tuple(queries), tuple(queries), cost, cost,
        "stagewise", tuple(adjusted),
    )


def optimize_naive(problem, target_success=1.0):
    """Find the best order by subset dynamic programming: O(k*2**k) updates.

    For S ending in i, cost(S)=(2*R+1)*cost(S without i)+(R+adjustment)*c[i].
    R depends only on these two subsets, which makes this recurrence exact.
    Memory is O(2**k), excluding the input cardinality table.
    """
    target = _target(target_success)
    length = len(problem.intersection_sizes)
    best = [math.inf] * length
    last = [-1] * length
    best[0] = 0.0
    evaluations = 0
    for mask in range(1, length):
        goal = target if mask == length - 1 else 1.0
        for i in range(problem.k):
            if not mask & (1 << i):
                continue
            previous = mask ^ (1 << i)
            ratio = problem.intersection_sizes[mask] / problem.intersection_sizes[previous]
            rounds, adjusted, _ = _amplification_plan(ratio, goal)
            value = (2 * rounds + 1) * best[previous] + (rounds + int(adjusted)) * problem.costs[i]
            evaluations += 1
            if value < best[mask]:
                best[mask], last[mask] = value, i
    if last[-1] < 0:
        raise ValueError("all orders have costs too large for floating-point evaluation")
    mask, reverse_order = length - 1, []
    while mask:
        i = last[mask]
        reverse_order.append(i)
        mask ^= 1 << i
    result = evaluate_naive(problem, tuple(reversed(reverse_order)), target)
    return replace(result, optimality="global_naive_model", evaluations=evaluations,
                   orders_searched=math.factorial(problem.k), orders_exhaustive=True)


def ss_success_probabilities(conditional_probabilities, iterations):
    """Exact SS stage probabilities; inputs are beta_i**2, NOT amplitudes beta_i."""
    probabilities = tuple(_probability(p, "conditional probability") for p in conditional_probabilities)
    iterations = tuple(_integer(t, "iteration count") for t in iterations)
    if not probabilities or len(probabilities) != len(iterations) or any(t < 0 for t in iterations):
        raise ValueError("provide equal nonzero lengths and nonnegative iteration counts")
    success, stages = 1.0, []
    for probability, rounds in zip(probabilities, iterations):
        success = _grover_probability(probability * success, rounds)
        stages.append(success)
    return tuple(stages)


def ss_query_counts(iterations):
    """Counts for one SS preparation, in nesting order, including inverses."""
    iterations = tuple(_integer(t, "iteration count") for t in iterations)
    if not iterations or any(t < 0 for t in iterations):
        raise ValueError("iterations must be a nonempty sequence of nonnegative integers")
    counts, suffix = [0] * len(iterations), 1
    for i in range(len(iterations) - 1, -1, -1):
        suffix *= 2 * iterations[i] + 1
        counts[i] = suffix
    return tuple(counts)


def evaluate_ss(problem, iterations, order=None, target_success=0.99, completion="amplify"):
    """Evaluate Appendix A's preparation and an explicitly chosen completion.

    'restart' has success 1 eventually and reports EXPECTED query counts C/P.
    'repeat' has a finite run count; success 1 is unavailable unless inner P=1.
    'none' reports the raw success even if it is below target_success.
    """
    target = _target(target_success)
    order = _order(problem, order)
    sizes = problem.prefix_sizes(order)
    iterations = tuple(_integer(t, "iteration count") for t in iterations)
    if len(iterations) != problem.k:
        raise ValueError("provide one iteration count per nesting level")
    conditional = tuple(sizes[i + 1] / sizes[i] for i in range(problem.k))
    stages = ss_success_probabilities(conditional, iterations)
    counts = [0] * problem.k
    for oracle, count in zip(order, ss_query_counts(iterations)):
        counts[oracle] = count
    inner = stages[-1]
    inner_cost = _weighted_cost(counts, problem.costs)
    outer, adjusted, repetitions = 0, False, 1
    if completion == "amplify":
        outer, adjusted, success = _amplification_plan(inner, target)
        factor = 2 * outer + 1
    elif completion == "repeat":
        if inner <= 0 or (target == 1 and inner < 1):
            raise ValueError("the requested success cannot be reached by finitely many runs")
        repetitions = 1 if inner >= target else max(1, math.ceil(math.log1p(-target) / math.log1p(-inner)))
        factor = repetitions
        success = 1.0 if inner == 1 else -math.expm1(repetitions * math.log1p(-inner))
    elif completion == "restart":
        if inner <= 0:
            raise ValueError("a zero-success preparation cannot succeed by restarting")
        factor = repetitions = 1.0 / inner
        success = 1.0
    elif completion == "none":
        factor, success = 1, inner
    else:
        raise ValueError("completion must be 'amplify', 'repeat', 'restart', or 'none'")
    completed_counts = tuple(factor * q for q in counts)
    return NestedResult(
        "schrottenloher_stevens", order, sizes, iterations, stages, inner, success,
        target, tuple(counts), completed_counts, inner_cost,
        _weighted_cost(completed_counts, problem.costs), completion,
        (False,) * problem.k, outer, adjusted, repetitions, problem.k,
    )


def lemma11_iteration_bounds(upper_probabilities):
    """Appendix A.5's conservative caps from upper bounds on beta_i**2.

    A zero upper probability is assigned cap zero, since no amplification can
    help that impossible filter. Exact-data optimizers need not obey these caps.
    """
    limits = []
    for p in upper_probabilities:
        p = _probability(p, "upper conditional probability")
        if p == 0:
            limits.append(0)
        else:
            bound = math.pi / (4 * math.asin(math.sqrt(p))) - 0.5
            limits.append(max(0, math.floor(bound + 16 * math.ulp(max(1.0, abs(bound))))))
    return tuple(limits)


def ss_success_bounds(lower_probabilities, upper_probabilities, iterations):
    """Lemma 11's valid interval, enforcing its sufficient iteration caps."""
    lower = tuple(_probability(p, "lower conditional probability") for p in lower_probabilities)
    upper = tuple(_probability(p, "upper conditional probability") for p in upper_probabilities)
    iterations = tuple(_integer(t, "iteration count") for t in iterations)
    if not lower or len(lower) != len(upper) or len(lower) != len(iterations):
        raise ValueError("bounds and iterations must have the same nonzero length")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("lower probabilities must not exceed upper probabilities")
    caps = lemma11_iteration_bounds(upper)
    if any(t < 0 or t > cap for t, cap in zip(iterations, caps)):
        raise ValueError("iteration counts exceed Lemma 11's sufficient bounds")
    return (ss_success_probabilities(lower, iterations)[-1],
            ss_success_probabilities(upper, iterations)[-1])


def _ss_continuous_stats(conditional, costs, iterations):
    """Continuous extension of Appendix A's formulas and analytic gradients.

    Returned gradients are with respect to the real iteration counts. This is
    a relaxation for optimization; noninteger counts do not define a circuit.
    """
    k = len(iterations)
    odd = tuple(2 * t + 1 for t in iterations)
    contributions = [0.0] * k
    suffix = 1.0
    for i in range(k - 1, -1, -1):
        suffix *= odd[i]
        contributions[i] = costs[i] * suffix
    cost = math.fsum(contributions)
    cost_gradient, accumulated = [], 0.0
    for factor, contribution in zip(odd, contributions):
        accumulated += contribution
        cost_gradient.append(2 * accumulated / factor)
    probability, probability_gradient = 1.0, [0.0] * k
    for i, (beta2, factor) in enumerate(zip(conditional, odd)):
        incoming = min(1.0, max(0.0, beta2 * probability))
        theta = math.asin(math.sqrt(incoming))
        angle = factor * theta
        sine = math.sin(2 * angle)
        if factor == 1.0:
            probability = incoming
            sensitivity = beta2
        else:
            probability = math.sin(angle) ** 2
            denominator = 2 * math.sqrt(incoming * (1 - incoming))
            if denominator == 0:
                # With Lemma 11 bounds, incoming=1 implies factor=1, handled
                # above. The other endpoint is the small-probability limit.
                sensitivity = beta2 * factor * factor
            else:
                sensitivity = beta2 * factor * sine / denominator
        for j in range(i):
            probability_gradient[j] *= sensitivity
        probability_gradient[i] = 2 * theta * sine
    return cost, probability, tuple(cost_gradient), tuple(probability_gradient)


def _ss_continuous_bounds(conditional):
    bounds = []
    for beta2 in conditional:
        bound = max(0.0, math.pi / (4 * math.asin(math.sqrt(beta2))) - 0.5)
        nearest = round(bound)
        # Keep mathematically integral bounds (e.g. beta^2=1/4) integral.
        if abs(bound - nearest) <= 16 * math.ulp(max(1.0, abs(bound))):
            bound = float(nearest)
        bounds.append(bound)
    return tuple(bounds)


def _ss_amplify_relaxation(conditional, costs, iterations, target):
    """Allow a real outer multiplier; integer circuits are evaluated separately."""
    cost, probability, dc, dp = _ss_continuous_stats(conditional, costs, iterations)
    if probability <= 0:
        raise ValueError("a zero-success preparation cannot be amplified")
    if probability >= target:
        return cost, probability, dc
    theta = math.asin(math.sqrt(probability))
    multiplier = math.asin(math.sqrt(target)) / theta
    sensitivity = 1 / (2 * theta * math.sqrt(probability * (1 - probability)))
    gradient = tuple(multiplier * (c - cost * sensitivity * p) for c, p in zip(dc, dp))
    return multiplier * cost, probability, gradient


def _ss_integer_amplify_search(problem, order, target, caps, continuous):
    """Compare actual amplified costs, exhaustively when the bounds are small.

    For larger spaces, refine integer candidates one coordinate at a time.
    Besides rounding and boundary candidates, try counts at which the required
    number of final amplification rounds decreases. These thresholds follow by
    inverting the monotone SS recurrence within the retained Lemma 11 bounds.
    This is a deterministic local search, not a global-optimization theorem.
    """
    cache = {}

    def evaluate(counts):
        counts = tuple(counts)
        if counts not in cache:
            cache[counts] = evaluate_ss(problem, counts, order, target, completion="amplify")
        return cache[counts]

    def key(result):
        return result.weighted_cost, result.outer_iterations, result.iterations

    if math.prod(cap + 1 for cap in caps) <= 10000:
        best = min((evaluate(t) for t in itertools.product(*(range(cap + 1) for cap in caps))), key=key)
        return best, len(cache), True

    floor = tuple(min(cap, max(0, math.floor(t))) for cap, t in zip(caps, continuous))
    ceil = tuple(min(cap, max(0, math.ceil(t))) for cap, t in zip(caps, continuous))
    seeds = {floor, ceil, tuple(caps), (0,) * problem.k}
    if problem.k <= 8:
        seeds.update(itertools.product(*(sorted({lo, hi}) for lo, hi in zip(floor, ceil))))
    best = min((evaluate(t) for t in sorted(seeds)), key=key)
    sizes = problem.prefix_sizes(order)
    conditional = tuple(sizes[i + 1] / sizes[i] for i in range(problem.k))
    angle_target = math.asin(math.sqrt(target))
    for _ in range(20):
        previous = best.iterations
        for i, cap in enumerate(caps):
            if cap == 0:
                continue
            counts = best.iterations
            candidates = {0, cap, floor[i], ceil[i], counts[i]}
            candidates.update(t for t in (counts[i] - 1, counts[i] + 1) if 0 <= t <= cap)
            zero = list(counts)
            zero[i] = 0
            max_outer = evaluate(zero).outer_iterations
            rounds = set(range(min(16, max_outer) + 1))
            rounds.update(r for r in range(max(0, best.outer_iterations - 2),
                                           min(max_outer, best.outer_iterations + 2) + 1))
            # Cover large outer-round ranges without enumerating O(sqrt(N)) values.
            r = 1
            while r < max_outer:
                rounds.add(r)
                r *= 2
            rounds.add(max_outer)
            before = 1.0 if i == 0 else best.stage_success_probabilities[i - 1]
            incoming_angle = math.asin(math.sqrt(conditional[i] * before))
            for r in sorted(rounds):
                required = math.sin(angle_target / (2 * r + 1)) ** 2
                for j in range(problem.k - 1, i, -1):
                    required = (math.sin(math.asin(math.sqrt(min(1.0, required)))
                                         / (2 * counts[j] + 1)) ** 2 / conditional[j])
                    if required > 1 + 32 * math.ulp(1.0):
                        break
                if required > 1 + 32 * math.ulp(1.0):
                    continue
                threshold = (math.asin(math.sqrt(min(1.0, required))) / incoming_angle - 1) / 2
                boundary = math.ceil(threshold)
                candidates.update(t for t in (boundary - 1, boundary, boundary + 1) if 0 <= t <= cap)
            for t in sorted(candidates):
                trial = list(counts)
                trial[i] = t
                result = evaluate(trial)
                if key(result) < key(best):
                    best = result
        if best.iterations == previous:
            break
    return best, len(cache), False


def optimize_ss(problem, target_success=0.99, maxiter=50000):
    """Minimize the final AMPLIFIED cost at the naive-optimal, unmerged order.

    First run optimize_naive, then optimize_ss_for_order using its order.
    This standalone function includes BOTH parts. To time them separately,
    call optimize_naive once and pass its order to optimize_ss_for_order.
    """
    target = _target(target_success)
    maxiter = _integer(maxiter, "maxiter")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    reference = optimize_naive(problem, target)
    result = optimize_ss_for_order(problem, reference.order, target, maxiter)
    return replace(result, naive_target_success=target)


def optimize_ss_for_order(problem, order, target_success=0.99, maxiter=50000):
    """Optimize amplified SS for an explicit order, WITHOUT an order search.

    Every oracle is one level. To reuse the naive-optimal order, supply
    optimize_naive(problem, target_success).order. The caller accounts for
    obtaining that order; this function performs only the additional SS work.
    For integer inner counts t, the objective is (2*R(t)+1)*C(t), where R(t)
    is exactly the outer round count used by evaluate_ss(..., completion='amplify').
    Returned query counts and weighted_cost are finite circuit costs.

    Retain Appendix A.5 / Lemma 11's bounds on the inner counts. For initial
    candidates, L-BFGS-B minimizes the continuous relaxation
      C(t)*max(1, asin(sqrt(target_success))/asin(sqrt(P(t)))).
    This allows a real final amplification multiplier; it is NOT the reported
    circuit cost and is NOT the C/P restart objective. The solver starts at
    the upper bounds, using scaled coordinates and analytic gradients. If its
    line search fails at the target-probability kink, a bounded Powell solve
    supplies candidates instead. Failure of both solvers raises RuntimeError.
    maxiter limits its iterations, not the subsequent integer evaluations.

    At most 10000 bounded integer combinations are enumerated in full.
    Otherwise rounded solutions and boundary candidates are refined by local
    integer search using the actual amplified cost. 'best_found' carries no
    global guarantee. 'global_fixed_order_bounds' certifies only the chosen
    order and these integer bounds, not all SS orders or possible iterations.
    continuous_amplified_cost reports the relaxation, continuous_expected_cost
    is unused (None), and rounding_cost_ratio compares the returned integer
    cost with the continuous value, including any integer refinement.
    """
    target = _target(target_success)
    maxiter = _integer(maxiter, "maxiter")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    order = _order(problem, order)
    sizes = problem.prefix_sizes(order)
    conditional = tuple(sizes[i + 1] / sizes[i] for i in range(problem.k))
    costs = tuple(problem.costs[i] for i in order)
    upper = _ss_continuous_bounds(conditional)
    active = tuple(i for i, bound in enumerate(upper) if bound > 0)
    scale, _, _ = _ss_amplify_relaxation(conditional, costs, upper, target)
    if not math.isfinite(scale):
        raise ValueError("initial amplified cost exceeds floating-point range")
    evaluations = 1
    solver_method = "L-BFGS-B"
    if active:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError(
                "optimize_ss requires SciPy; run python -m pip install -r requirements.txt"
            ) from exc

        def objective(coordinates):
            nonlocal evaluations
            evaluations += 1
            counts = [0.0] * problem.k
            for i, coordinate in zip(active, coordinates):
                counts[i] = upper[i] * float(coordinate)
            value, _, gradient = _ss_amplify_relaxation(conditional, costs, counts, target)
            if not math.isfinite(value):
                raise ValueError("continuous amplified cost exceeds floating-point range")
            return value / scale, [gradient[i] * upper[i] / scale for i in active]

        solution = minimize(
            objective, [1.0] * len(active), method="L-BFGS-B", jac=True,
            bounds=[(0.0, 1.0)] * len(active),
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        solver_iterations = int(getattr(solution, "nit", 0))
        message = str(solution.message)
        if not solution.success:
            # The relaxation has a kink when the inner success reaches target.
            # A derivative-free bounded solve handles this without accepting a
            # failed gradient solve as a successfully optimized point.
            start = [min(1.0, max(0.0, float(x)))
                     for x in getattr(solution, "x", [1.0] * len(active))]
            solution = minimize(
                lambda x: objective(x)[0], start, method="Powell",
                bounds=[(0.0, 1.0)] * len(active),
                options={"maxiter": maxiter, "xtol": 1e-9, "ftol": 1e-12},
            )
            solver_method += "+Powell"
            message += "; Powell: " + str(solution.message)
            solver_iterations += int(getattr(solution, "nit", 0))
            if not solution.success:
                raise RuntimeError("SS continuous optimization failed: " + message)
        continuous = [0.0] * problem.k
        for i, coordinate in zip(active, solution.x):
            continuous[i] = min(upper[i], max(0.0, upper[i] * float(coordinate)))
    else:
        continuous = [0.0] * problem.k
        message, solver_iterations = "All Lemma 11 iteration bounds are zero.", 0
    continuous = tuple(continuous)
    relaxed_cost, relaxed_probability, _ = _ss_amplify_relaxation(conditional, costs, continuous, target)
    evaluations += 1
    ordered_caps = tuple(math.floor(bound) for bound in upper)
    result, integer_evaluations, exhaustive = _ss_integer_amplify_search(
        problem, order, target, ordered_caps, continuous)
    evaluations += integer_evaluations
    caps = [0] * problem.k
    for oracle, bound in zip(order, upper):
        caps[oracle] = math.floor(bound)
    return replace(
        result, optimality="global_fixed_order_bounds" if exhaustive else "best_found",
        evaluations=evaluations, orders_searched=1,
        orders_exhaustive=(problem.k == 1), iteration_bounds=tuple(caps),
        optimization_method=solver_method + "_relaxation_then_integer_amplify", optimizer_success=True,
        optimizer_message=message, optimizer_iterations=solver_iterations,
        continuous_iterations=continuous,
        continuous_success_probability=relaxed_probability,
        continuous_amplified_cost=relaxed_cost,
        rounding_cost_ratio=result.weighted_cost / relaxed_cost,
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=float, default=0.99,
                        help="common success target for naive nesting and amplified SS (default: 0.99)")
    parser.add_argument("--cost-ratio", type=float, default=10.0)
    parser.add_argument("--maxiter", type=int, default=50000)
    args = parser.parse_args()
    problem = SearchProblem([2**40, 1000, 100, 1], [1, args.cost_ratio])
    results = [optimize_naive(problem, args.target),
               optimize_ss(problem, args.target, maxiter=args.maxiter)]
    print(json.dumps([result.as_dict() for result in results], indent=2, allow_nan=False))


if __name__ == "__main__":
    _main()
