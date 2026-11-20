package genetic

import (
	"encoding/json"
	"math/rand/v2"
	"reflect"
	"slices"
	"sync"
	"testing"
)

func newTestEngine(t *testing.T, cfg StreamingConfig) *StreamingEngine[[]float64, float64] {
	t.Helper()
	bounds := []ParameterBounds{{Min: 0, Max: 1}}
	e := NewStreamingEngine[[]float64, float64](
		func(rng *rand.Rand) []float64 { return []float64{rng.Float64()} },
		&TournamentSelector[[]float64, float64]{TournamentSize: 3},
		&UniformCombiner[[]float64, float64, float64]{MixProbability: 0.5},
		&BoundedPerturbator{Bounds: bounds},
		SliceCloner[[]float64, float64]{},
		cfg,
	)
	e.Start()
	return e
}

// Archive must stay sorted descending and never exceed PoolSize
func TestStreaming_ArchiveInvariant(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.PoolSize = 8
	cfg.Seed = 1
	e := newTestEngine(t, cfg)

	for i := range 500 {
		_, id := e.Propose()
		e.CompleteEvaluation(id, float64(i%37))
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.archive) != cfg.PoolSize {
		t.Fatalf("archive size %d", len(e.archive))
	}
	for i := 1; i < len(e.archive); i++ {
		if e.archive[i-1].Score < e.archive[i].Score {
			t.Fatalf("unsorted at %d", i)
		}
	}
}

// Elites must survive an unbounded stream of inferior outcomes
func TestStreaming_Elitism(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.PoolSize = 8
	cfg.Seed = 2
	e := newTestEngine(t, cfg)

	_, id := e.Propose()
	e.CompleteEvaluation(id, 1000)

	for range 1000 {
		_, id := e.Propose()
		e.CompleteEvaluation(id, 0.1)
	}
	if best := e.Stats().BestScore; best != 1000 {
		t.Fatalf("elite lost: %v", best)
	}
}

// Under SetEqual a genotype is archived once, at its latest score: a repeat
// outcome moves the member down or up to where it now sorts, while a distinct
// genotype with an equal score is still admitted
func TestStreaming_ARepeatOutcomeReplacesItsGenotypesScore(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.PoolSize = 8
	e := newTestEngine(t, cfg)
	e.SetEqual(slices.Equal[[]float64])

	for i := range 64 {
		e.CompleteEvaluation(e.BeginEvaluation([]float64{0.5}), 1)
		e.CompleteEvaluation(e.BeginEvaluation([]float64{float64(i % 3)}), 1)
	}
	e.CompleteEvaluation(e.BeginEvaluation([]float64{0.5}), 0.25)
	e.CompleteEvaluation(e.BeginEvaluation([]float64{2}), 3)

	members := e.Snapshot().Members
	if len(members) != 4 {
		t.Fatalf("archive holds %d members, want 4 distinct", len(members))
	}
	for i := range members {
		for j := i + 1; j < len(members); j++ {
			if slices.Equal(members[i].Data, members[j].Data) {
				t.Fatalf("duplicate genotype %v", members[i].Data)
			}
		}
	}
	if members[0].Score != 3 || members[0].Data[0] != 2 {
		t.Fatalf("the higher repeat did not move its genotype up: %v", members[0])
	}
	if members[3].Score != 0.25 || members[3].Data[0] != 0.5 {
		t.Fatalf("the lower repeat did not move its genotype down: %v", members[3])
	}
}

// Under SetMerge a repeat outcome folds into the score its genotype holds and
// is counted, the member moving to where the merged score sorts
func TestStreaming_ARepeatOutcomeMergesIntoItsGenotypesScore(t *testing.T) {
	e := newTestEngine(t, DefaultStreamingConfig())
	e.SetEqual(slices.Equal[[]float64])
	e.SetMerge(MeanScore[float64])

	e.CompleteEvaluation(e.BeginEvaluation([]float64{0.25}), 3.5)
	for _, score := range []float64{6, 2, 1} {
		e.CompleteEvaluation(e.BeginEvaluation([]float64{0.5}), score)
	}

	want := []Candidate[[]float64, float64]{
		{Data: []float64{0.25}, Score: 3.5, Count: 1},
		{Data: []float64{0.5}, Score: 3, Count: 3},
	}
	if got := e.Snapshot().Members; !reflect.DeepEqual(got, want) {
		t.Fatalf("archive %v, want %v", got, want)
	}
}

// A rank that keys a member by its score keeps the archive an engine with no
// rank keeps, to the checkpoint, through outcomes that merge, displace and fail
// to enter
func TestStreaming_ARankByTheScoreChangesNothing(t *testing.T) {
	run := func(ranked bool) StreamingState[[]float64, float64] {
		cfg := DefaultStreamingConfig()
		cfg.PoolSize, cfg.Seed = 8, 9
		e := newTestEngine(t, cfg)
		e.SetEqual(slices.Equal[[]float64])
		e.SetMerge(MeanScore[float64])
		if ranked {
			e.SetRank(func(members []Candidate[[]float64, float64], keys []float64) {
				for i, c := range members {
					keys[i] = c.Score
				}
			})
		}
		for i := range 400 {
			g, id := e.Propose()
			if i%3 == 2 {
				e.AbandonEvaluation(id)
				g, id = e.Reevaluate()
			}
			e.CompleteEvaluation(id, float64(i*7919%13)+g[0])
		}
		st, err := e.Checkpoint()
		if err != nil {
			t.Fatal(err)
		}
		return st
	}
	if plain, ranked := run(false), run(true); !reflect.DeepEqual(plain, ranked) {
		t.Fatal("an archive ranked by its scores is not the one an engine with no rank keeps")
	}
}

// Under SetRank the keys order the archive and the last of a full one leaves:
// a member four outcomes have tested stays over a newcomer's higher score, and
// an entrant whose key is the lowest does not enter. Best and Stats follow
func TestStreaming_ARankKeepsWhatItsKeysPutFirst(t *testing.T) {
	e := rankedByCount(t)
	want := []Candidate[[]float64, float64]{
		{Data: []float64{0.1}, Score: 6, Count: 4},
		{Data: []float64{0.3}, Score: 9, Count: 1},
	}
	if got := e.Snapshot().Members; !reflect.DeepEqual(got, want) {
		t.Fatalf("archive %v, want %v", got, want)
	}
	best, _ := e.Best()
	if st := e.Stats(); best.Score != 6 || st.BestScore != 6 || st.WorstScore != 9 {
		t.Fatalf("best %v, and stats from %v to %v: want the first and the last by rank, 6 and 9", best.Score, st.BestScore, st.WorstScore)
	}
}

// rankedByCount is an engine of two members ranked by score x count/(count+2):
// one scored 6 four times, key 4; one scored 3, key 1, which one scored 9, key
// 3, displaces; and one scored 8, key 2.67, under both
func rankedByCount(t *testing.T) *StreamingEngine[[]float64, float64] {
	t.Helper()
	cfg := DefaultStreamingConfig()
	cfg.PoolSize = 2
	e := newTestEngine(t, cfg)
	e.SetEqual(slices.Equal[[]float64])
	e.SetMerge(MeanScore[float64])
	e.SetRank(func(members []Candidate[[]float64, float64], keys []float64) {
		for i, c := range members {
			keys[i] = c.Score * float64(c.Count) / float64(c.Count+2)
		}
	})
	for _, o := range []struct{ gene, score float64 }{{0.1, 6}, {0.1, 6}, {0.1, 6}, {0.1, 6}, {0.2, 3}, {0.3, 9}, {0.4, 8}} {
		e.CompleteEvaluation(e.BeginEvaluation([]float64{o.gene}), o.score)
	}
	return e
}

// A ranked archive is not in score order, and its keys are its host's: Restore
// takes it as it stands into an engine that has a rank, and an engine with
// none still refuses it
func TestStreaming_ARankedArchiveRestoresInItsOrder(t *testing.T) {
	source := rankedByCount(t)
	state, err := source.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	if err := rankedByCount(t).Restore(state); err != nil {
		t.Fatalf("an engine with a rank refused a ranked archive: %v", err)
	}
	cfg := DefaultStreamingConfig()
	cfg.PoolSize = 2
	if err := newTestEngine(t, cfg).Restore(state); err == nil {
		t.Fatal("an engine with no rank took an archive out of score order")
	}
}

// Reevaluate hands out the member scored or handed out longest ago, the better
// of two that tie, or the one a SetDue hook names, as a copy of the caller's
// own; with no SetEqual to know a repeat by, where the hook names none, or
// while stopped, it hands out nothing
func TestStreaming_ReevaluateHandsOutTheMemberDue(t *testing.T) {
	e := newTestEngine(t, DefaultStreamingConfig())
	for _, g := range []float64{0.3, 0.1, 0.2} {
		e.CompleteEvaluation(e.BeginEvaluation([]float64{g}), g)
	}
	if _, id := e.Reevaluate(); id != 0 {
		t.Fatal("a member was handed out with no SetEqual")
	}
	e.SetEqual(slices.Equal[[]float64])

	var got []float64
	var first EvalID
	for i := range 6 {
		g, id := e.Reevaluate()
		got = append(got, g[0])
		g[0] = -1
		switch i {
		case 0:
			first = id
		case 2:
			// Scored as late as the third was handed out, and the worse of the two
			e.CompleteEvaluation(first, 0.05)
		}
	}
	if want := []float64{0.3, 0.1, 0.2, 0.1, 0.2, 0.3}; !slices.Equal(got, want) {
		t.Fatalf("handed out %v, want %v", got, want)
	}
	want := []Candidate[[]float64, float64]{
		{Data: []float64{0.2}, Score: 0.2, Count: 1},
		{Data: []float64{0.1}, Score: 0.1, Count: 1},
		{Data: []float64{0.3}, Score: 0.05, Count: 1},
	}
	if members := e.Snapshot().Members; !reflect.DeepEqual(members, want) {
		t.Fatalf("archive %v, want %v", members, want)
	}

	e.SetDue(func(members []Candidate[[]float64, float64]) int { return len(members) - 1 })
	if g, _ := e.Reevaluate(); g[0] != 0.3 {
		t.Fatalf("handed out %v, want the last member, which the host named", g)
	}
	e.SetDue(func([]Candidate[[]float64, float64]) int { return -1 })
	if _, id := e.Reevaluate(); id != 0 {
		t.Fatal("a member was handed out where the host named none")
	}

	e.Stop()
	if _, id := e.Reevaluate(); id != 0 {
		t.Fatal("a member was handed out while stopped")
	}
}

// A schedule's knobs breed the offspring made once a generation has ended, the
// configuration's those before, and a checkpoint carries the pair in force
func TestStreaming_AScheduleSetsTheKnobsLaterOffspringAreBredWith(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.ProposalCapacity = 2
	cfg.MinOutcomesPerGen = 4
	var bred []float64
	e := NewStreamingEngine[[]float64, float64](
		func(rng *rand.Rand) []float64 { return []float64{rng.Float64()} },
		&TournamentSelector[[]float64, float64]{TournamentSize: 3},
		&UniformCombiner[[]float64, float64, float64]{MixProbability: 0.5},
		perturbFunc(func(_ *[]float64, rate, strength float64, _ *rand.Rand) { bred = append(bred, rate*strength) }),
		SliceCloner[[]float64, float64]{},
		cfg,
	)
	e.SetSchedule(func(gen int, _ PoolStats[float64]) (float64, float64) { return 0.5, 1 / float64(gen+1) })
	e.Start()

	for range 12 {
		g, id := e.Propose()
		e.CompleteEvaluation(id, g[0])
	}
	want := []float64{0.2 * 0.15, 0.2 * 0.15, 0.5 / 2, 0.5 / 2, 0.5 / 2, 0.5 / 2, 0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5 / 4, 0.5 / 4}
	if !reflect.DeepEqual(bred, want) {
		t.Fatalf("offspring bred with %v, want %v", bred, want)
	}
	if st, _ := e.Checkpoint(); st.Rate != 0.5 || st.Strength != 0.25 {
		t.Fatalf("checkpoint carries rate %v and strength %v", st.Rate, st.Strength)
	}
}

type perturbFunc func(solution *[]float64, rate, strength float64, rng *rand.Rand)

func (f perturbFunc) Perturb(solution *[]float64, rate, strength float64, rng *rand.Rand) {
	f(solution, rate, strength, rng)
}

// An evaluation in flight is lost only to a full table, the oldest first: fifteen
// held through any number of completed issues stay, the table never holds more
// than its capacity, the outcome of one it gave up is ignored, and a full table
// with an id far from its home slot restores
func TestStreaming_AnEvaluationIsEvictedOnlyFromAFullTableTheOldestFirst(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.PendingCapacity = 16
	cfg.Seed = 6
	e := newTestEngine(t, cfg)

	var held []EvalID
	for range cfg.PendingCapacity - 1 {
		_, id := e.Propose()
		held = append(held, id)
	}
	// One past a lap of the table, so the id that fills it lands a lap from its home
	for range 10_001 {
		g, id := e.Propose()
		e.CompleteEvaluation(id, g[0])
	}
	if s := e.Stats(); s.Evicted != 0 || s.Pending != len(held) {
		t.Fatalf("evaluations in flight under the capacity were lost: evicted %d, pending %d", s.Evicted, s.Pending)
	}
	e.Propose()
	if st, _ := e.Checkpoint(); newTestEngine(t, cfg).Restore(st) != nil {
		t.Fatal("a full table, an id a lap from its home, did not restore")
	}
	e.Propose()
	if s := e.Stats(); s.Evicted != 1 || s.Pending != cfg.PendingCapacity {
		t.Fatalf("a put into a full table evicted %d and left %d pending, want one and the capacity", s.Evicted, s.Pending)
	}
	e.CompleteEvaluation(held[0], 3)
	if best := e.Stats().BestScore; best == 3 {
		t.Fatal("the oldest evaluation was not the one evicted")
	}
	e.CompleteEvaluation(held[1], 2)
	if best := e.Stats().BestScore; best != 2 {
		t.Fatalf("the second oldest was lost with the oldest: best %v", best)
	}
}

// Solutions handed out must not alias engine memory
func TestStreaming_NoAliasing(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.PoolSize = 4
	cfg.Seed = 4
	e := newTestEngine(t, cfg)

	held := make([][]float64, 0, 64)
	for range 64 {
		g, id := e.Propose()
		held = append(held, g)
		e.CompleteEvaluation(id, g[0])
	}

	// Mutating caller copies must not corrupt the archive
	for _, g := range held {
		g[0] = -1
	}
	if e.Stats().BestScore < 0 {
		t.Fatal("archive aliased caller memory")
	}
	for _, c := range e.Snapshot().Members {
		if c.Data[0] == -1 {
			t.Fatal("snapshot aliased caller memory")
		}
	}
}

// Concurrent Propose/Complete must be race-free (run with -race)
func TestStreaming_Concurrent(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.Seed = 5
	e := newTestEngine(t, cfg)

	var wg sync.WaitGroup

	for range 8 {
		wg.Go(func() {
			for range 2000 {
				g, id := e.Propose()
				if id != 0 {
					e.CompleteEvaluation(id, g[0])
				}
				_ = e.Stats()
			}
		})
	}

	wg.Wait()
}

// Identical seeds determine the complete proposal stream. The nanosecond budget
// is deliberately impossible: deterministic refill is the default and never
// consults it.
func TestStreaming_Deterministic(t *testing.T) {
	run := func() ([]float64, StreamingState[[]float64, float64]) {
		cfg := DefaultStreamingConfig()
		cfg.Seed = 0xDECAFBAD
		cfg.TickBudget = 1
		e := newTestEngine(t, cfg)
		stream := make([]float64, 0, 300)
		for range 300 {
			g, id := e.Propose()
			stream = append(stream, g[0])
			e.CompleteEvaluation(id, g[0])
		}
		state, err := e.Checkpoint()
		if err != nil {
			t.Fatal(err)
		}
		return stream, state
	}
	aStream, aState := run()
	bStream, bState := run()
	if !reflect.DeepEqual(aStream, bStream) {
		t.Fatal("identical seeds produced different proposal streams")
	}
	if !reflect.DeepEqual(aState, bState) {
		t.Fatal("identical seeded operation sequences produced different states")
	}
}

// A checkpoint carries what a snapshot of the archive alone would lose: PCG
// position, queued offspring, pending genotypes, partial-generation outcomes
// and the next evaluation id, so the restored engine continues the exact stream
func TestStreaming_CheckpointContinuesTheExactStream(t *testing.T) {
	cfg := DefaultStreamingConfig()
	cfg.Seed = 0x51A7E
	cfg.PoolSize = 8
	cfg.ProposalCapacity = 8
	cfg.PendingCapacity = 16
	cfg.MinOutcomesPerGen = 3

	// A merge, a schedule and re-evaluations put counts, seen marks and knobs
	// other than the configuration's in the state
	build := func() *StreamingEngine[[]float64, float64] {
		e := newTestEngine(t, cfg)
		e.SetEqual(slices.Equal[[]float64])
		e.SetMerge(MeanScore[float64])
		e.SetSchedule(func(gen int, st PoolStats[float64]) (float64, float64) { return 0.5, 0.3 / float64(gen%4+1) })
		return e
	}
	next := func(e *StreamingEngine[[]float64, float64], i int) ([]float64, EvalID) {
		if i%5 == 4 {
			if g, id := e.Reevaluate(); id != 0 {
				return g, id
			}
		}
		return e.Propose()
	}

	source := build()
	for i := range 15 {
		g, id := next(source, i)
		switch i % 4 {
		case 0, 1:
			source.CompleteEvaluation(id, g[0]+float64(i)/100)
		case 2:
			source.AbandonEvaluation(id)
		case 3:
			// Deliberately leave work in flight.
		}
	}

	checkpoint, err := source.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	var decoded StreamingState[[]float64, float64]
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}

	restored := build()
	for range 5 { // Prove Restore replaces receiver-local state.
		g, id := restored.Propose()
		restored.CompleteEvaluation(id, -g[0])
	}
	if err := restored.Restore(decoded); err != nil {
		t.Fatal(err)
	}
	got, err := restored.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(checkpoint, got) {
		t.Fatal("restored engine does not equal its checkpoint")
	}

	// Resolve the evaluations that crossed the checkpoint, then compare a long
	// mixture of completion and abandonment. IDs and genotypes must agree at every
	// issuance, not merely converge on the same best score.
	for _, pending := range checkpoint.Pending {
		score := pending.Data[0]
		source.CompleteEvaluation(pending.ID, score)
		restored.CompleteEvaluation(pending.ID, score)
	}
	for i := range 250 {
		a, aid := next(source, i)
		b, bid := next(restored, i)
		if aid != bid || !reflect.DeepEqual(a, b) {
			t.Fatalf("proposal %d differs: (%d, %v) != (%d, %v)", i, aid, a, bid, b)
		}
		if i%7 == 0 {
			source.AbandonEvaluation(aid)
			restored.AbandonEvaluation(bid)
		} else {
			score := a[0] + float64(i%11)/100
			source.CompleteEvaluation(aid, score)
			restored.CompleteEvaluation(bid, score)
		}
	}
	aState, err := source.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	bState, err := restored.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(aState, bState) {
		t.Fatal("restored engine diverged after identical operations")
	}
}

func TestStreaming_RestoreRejectsInvalidStateWithoutMutation(t *testing.T) {
	e := newTestEngine(t, DefaultStreamingConfig())
	e.CompleteEvaluation(e.BeginEvaluation([]float64{0.5}), 1)
	e.Propose()
	before, err := e.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	badVersion := before
	badVersion.Version++
	badConfig := before
	badConfig.Config.PoolSize++
	badSlot := before
	badSlot.Pending = slices.Clone(before.Pending)
	badSlot.Pending[0].Slot = len(e.pending.ids)
	badSeen := before
	badSeen.Seen = nil
	badCount := before
	badCount.Archive = slices.Clone(before.Archive)
	badCount.Archive[0].Count = 0
	for name, bad := range map[string]StreamingState[[]float64, float64]{
		"version": badVersion,
		"config":  badConfig,
		"slot":    badSlot,
		"seen":    badSeen,
		"count":   badCount,
	} {
		t.Run(name, func(t *testing.T) {
			if err := e.Restore(bad); err == nil {
				t.Fatal("invalid checkpoint was accepted")
			}
			after, err := e.Checkpoint()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before, after) {
				t.Fatal("failed restore changed the engine")
			}
		})
	}
}

// Stop then Start must resume proposal issuance
func TestStreaming_Restart(t *testing.T) {
	e := newTestEngine(t, DefaultStreamingConfig())
	e.Stop()
	if _, id := e.Propose(); id != 0 {
		t.Fatal("expected zero id while stopped")
	}
	e.Start()
	if _, id := e.Propose(); id == 0 {
		t.Fatal("engine did not restart")
	}
}

func BenchmarkStreaming_ProposeComplete(b *testing.B) {
	cfg := DefaultStreamingConfig()
	cfg.Seed = 7
	bounds := []ParameterBounds{{Min: 0, Max: 1}, {Min: 0, Max: 1}}
	e := NewStreamingEngine[[]float64, float64](
		func(rng *rand.Rand) []float64 { return []float64{rng.Float64(), rng.Float64()} },
		&TournamentSelector[[]float64, float64]{TournamentSize: 3},
		&UniformCombiner[[]float64, float64, float64]{MixProbability: 0.5},
		&BoundedPerturbator{Bounds: bounds},
		SliceCloner[[]float64, float64]{},
		cfg,
	)
	e.Start()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; b.Loop(); i++ {
		g, id := e.Propose()
		e.CompleteEvaluation(id, g[0])
	}
}
