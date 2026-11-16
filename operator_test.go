package genetic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Mutation must never leave the declared bounds under any boundary mode
func FuzzBoundedPerturbator(f *testing.F) {
	f.Add(uint64(1), 0.5, 0.9, -3.0, 7.0, uint8(0))
	f.Add(uint64(2), 1.0, 2.0, 0.0, 1.0, uint8(1))
	f.Add(uint64(3), 1.0, 2.0, -1.0, 1.0, uint8(2))

	f.Fuzz(func(t *testing.T, seed uint64, rate, strength, lo, hi float64, mode uint8) {
		span := hi - lo
		if hi <= lo || span != span || span > math.MaxFloat64/2 ||
			rate < 0 || rate > 1 || strength < 0 || strength > 4 {
			t.Skip()
		}
		bp := &BoundedPerturbator{
			Bounds:   []ParameterBounds{{Min: lo, Max: hi}},
			Boundary: BoundaryMode(mode % 3),
		}
		rng := rand.New(rand.NewPCG(seed, seed))
		g := []float64{lo + (hi-lo)/2}

		for range 1000 {
			bp.Perturb(&g, rate, strength, rng)
			if g[0] < lo || g[0] > hi || g[0] != g[0] {
				t.Fatalf("out of bounds: %v not in [%v,%v] under mode %d", g[0], lo, hi, mode%3)
			}
		}
	})
}

// A selector must not allocate while the archive version holds
func TestSelectors_ZeroAllocOnASteadyVersion(t *testing.T) {
	members := make([]Candidate[[]float64, float64], 32)
	for i := range members {
		members[i] = Candidate[[]float64, float64]{Data: []float64{float64(i)}, Score: float64(i)}
	}
	dst := make([]Candidate[[]float64, float64], 2)
	rng := rand.New(rand.NewPCG(1, 2))

	for name, sel := range map[string]Selector[[]float64, float64]{
		"tournament": &TournamentSelector[[]float64, float64]{TournamentSize: 3},
		"roulette":   &RouletteSelector[[]float64, float64]{},
	} {
		sel.Select(members, dst, 1, rng)
		if n := testing.AllocsPerRun(1000, func() { sel.Select(members, dst, 1, rng) }); n != 0 {
			t.Fatalf("%s: expected 0 allocs, got %v", name, n)
		}
	}
}

func TestRouletteSelector_NegativeScores(t *testing.T) {
	members := []Candidate[[]float64, float64]{
		{Score: -10}, {Score: -5}, {Score: -1},
	}
	sel := &RouletteSelector[[]float64, float64]{}
	dst := make([]Candidate[[]float64, float64], 100)
	sel.Select(members, dst, 1, rand.New(rand.NewPCG(1, 2)))

	for _, c := range dst {
		if c.Score >= 0 {
			t.Fatalf("selected out-of-set candidate: %v", c.Score)
		}
	}
}

// A block goes to a child whole, from either parent, and an element no block
// names stays with its own parent
func TestBlockCombiner_KeepsABlockWithOneParent(t *testing.T) {
	parents := []Candidate[[]int, float64]{{Data: []int{1, 1, 1, 1, 1, 1}}, {Data: []int{2, 2, 2, 2, 2, 2}}}
	bc := &BlockCombiner[[]int, int, float64]{Blocks: [][]int{{0, 3, 4}, {1, 2}}, MixProbability: 0.5}
	rng := rand.New(rand.NewPCG(1, 2))
	dst := make([][]int, 2)
	seen := map[[2]int]bool{}
	for range 64 {
		if n := bc.Combine(parents, dst, rng); n != 2 {
			t.Fatalf("wrote %d offspring, want 2", n)
		}
		a, b := dst[0], dst[1]
		if a[3] != a[0] || a[4] != a[0] || a[2] != a[1] || a[5] != 1 || b[5] != 2 {
			t.Fatalf("a block was split or a free element moved: %v %v", a, b)
		}
		for i := range a[:5] {
			if a[i]+b[i] != 3 {
				t.Fatalf("the offspring are not complements: %v %v", a, b)
			}
		}
		seen[[2]int{a[0], a[1]}] = true
	}
	if len(seen) != 4 {
		t.Fatalf("the blocks mixed %d ways of 4", len(seen))
	}
}

// Bin decodes every bin centre back to its own bin, including the last
func TestParameterBounds_BinCenterRoundTrips(t *testing.T) {
	b := ParameterBounds{Min: -2, Max: 3}
	for n := 1; n <= 9; n++ {
		for i := range n {
			if got := b.Bin(b.BinCenter(i, n), n); got != i {
				t.Fatalf("bin %d of %d decoded as %d", i, n, got)
			}
		}
		if got := b.Bin(b.Max, n); got != n-1 {
			t.Fatalf("Max decoded as bin %d of %d", got, n)
		}
	}
}
