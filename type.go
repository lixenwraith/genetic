// Package genetic is an evolutionary optimizer for simulations and games: a
// caller-driven steady-state StreamingEngine whose evaluations complete
// asynchronously and checkpoint exactly, a generational Engine with a
// synchronous evaluator, and the operators both share. Standard library only.
package genetic

import "math/rand/v2"

// Solution is any candidate encoding
type Solution any

// Numeric constrains fitness score types
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// Candidate pairs an encoded solution with its evaluated score. Count is the
// outcomes a streaming archive's Score holds, 1 unless SetMerge folds repeats;
// the batch engine leaves it 0
type Candidate[S Solution, F Numeric] struct {
	Data  S
	Score F
	Count int
}

// Pool is a scored candidate set. Engines maintain Members sorted by Score
// descending, a streaming one under SetRank by its keys
type Pool[S Solution, F Numeric] struct {
	Members    []Candidate[S, F]
	Generation int
	Stats      PoolStats[F]
}

// PoolStats is an immutable engine state snapshot
type PoolStats[F Numeric] struct {
	BestScore    F
	WorstScore   F
	AverageScore F
	Diversity    float64
	Size         int
	Generation   int
	Pending      int
	Evaluations  uint64
	Evicted      uint64
}

type (
	InitializerFunc[S Solution]            func(rng *rand.Rand) S
	EvaluatorFunc[S Solution, F Numeric]   func(solution S) F
	TerminationFunc[S Solution, F Numeric] func(pool *Pool[S, F], iteration int) bool
	DiversityFunc[S Solution, F Numeric]   func(members []Candidate[S, F]) float64
	// EqualFunc reports whether two solutions are the same genotype; slices.Equal
	// instantiated on the slice type is the usual choice
	EqualFunc[S Solution] func(a, b S) bool
	// MergeFunc folds a repeat outcome into the score its genotype holds, old
	// over count outcomes; MeanScore is one
	MergeFunc[F Numeric] func(old, new F, count int) F
	// RankFunc writes the key each member of an archive ranks by, keys[i] for
	// members[i], the higher first
	RankFunc[S Solution, F Numeric] func(members []Candidate[S, F], keys []float64)
	// DueFunc names the archive member due another evaluation, by its place in
	// members; a place outside them names none
	DueFunc[S Solution, F Numeric] func(members []Candidate[S, F]) int
	// ScheduleFunc chooses the mutation rate and strength the next generation is
	// bred with, from the pool as the last one left it; AdaptiveStrength is one
	ScheduleFunc[F Numeric] func(generation int, stats PoolStats[F]) (rate, strength float64)
)

// Selector fills dst with parents drawn from members. version changes whenever
// members does, so a selector may keep work derived from them between calls.
// Implementations must not allocate on a steady version
type Selector[S Solution, F Numeric] interface {
	Select(members []Candidate[S, F], dst []Candidate[S, F], version uint64, rng *rand.Rand)
}

// Combiner writes offspring into dst, reusing dst buffers when shape-compatible.
// Returns the number of entries written
type Combiner[S Solution, F Numeric] interface {
	Combine(parents []Candidate[S, F], dst []S, rng *rand.Rand) int
}

// Perturbator mutates a solution in place.
// rate is the per-element probability, strength the magnitude scale
type Perturbator[S Solution] interface {
	Perturb(solution *S, rate, strength float64, rng *rand.Rand)
}

// Cloner copies src into dst, reusing dst capacity when possible.
// Engines require it to separate caller-owned from engine-owned solutions
type Cloner[S Solution] interface {
	Clone(dst, src S) S
}

// SliceCloner implements Cloner for slice-encoded genotypes
type SliceCloner[S ~[]T, T any] struct{}

func (SliceCloner[S, T]) Clone(dst, src S) S {
	if cap(dst) < len(src) {
		dst = make(S, len(src))
	}
	dst = dst[:len(src)]
	copy(dst, src)
	return dst
}
