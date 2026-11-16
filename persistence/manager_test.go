package persistence

import (
	"math/rand/v2"
	"reflect"
	"slices"
	"testing"

	"github.com/lixenwraith/genetic"
)

// A saved population comes back with its tag and each candidate's count, and
// an engine it is injected into holds the archive that was saved
func TestManager_APopulationKeepsItsTagAndCounts(t *testing.T) {
	build := func() *genetic.StreamingEngine[[]float64, float64] {
		e := genetic.NewStreamingEngine[[]float64, float64](
			func(rng *rand.Rand) []float64 { return []float64{rng.Float64()} },
			&genetic.TournamentSelector[[]float64, float64]{},
			&genetic.UniformCombiner[[]float64, float64, float64]{},
			&genetic.GaussianPerturbator[[]float64, float64]{},
			genetic.SliceCloner[[]float64, float64]{},
			genetic.DefaultStreamingConfig(),
		)
		e.SetEqual(slices.Equal[[]float64])
		e.SetMerge(genetic.MeanScore[float64])
		return e
	}
	source := build()
	for _, score := range []float64{4, 2} {
		source.CompleteEvaluation(source.BeginEvaluation([]float64{0.5}), score)
	}
	source.CompleteEvaluation(source.BeginEvaluation([]float64{0.25}), 1)

	store := NewManager(t.TempDir(), nil)
	dto := FromPool(source.Snapshot())
	dto.Tag = "16-6-2"
	if err := store.Save("net", dto); err != nil {
		t.Fatal(err)
	}
	loaded, err := store.Load("net")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Tag != "16-6-2" || loaded.Version != SchemaVersion {
		t.Fatalf("loaded tag %q at version %d", loaded.Tag, loaded.Version)
	}

	restored := build()
	restored.Inject(loaded.ToPool(), loaded.Generation)
	want := []genetic.Candidate[[]float64, float64]{
		{Data: []float64{0.5}, Score: 3, Count: 2},
		{Data: []float64{0.25}, Score: 1, Count: 1},
	}
	if got := restored.Snapshot().Members; !reflect.DeepEqual(got, want) {
		t.Fatalf("restored archive %v, want %v", got, want)
	}
}
