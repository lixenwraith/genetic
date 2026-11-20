package genetic

import (
	"math/rand/v2"
	"sync"
	"sync/atomic"
	"time"
)

// EvalID identifies an in-flight evaluation; 0 is never issued
type EvalID uint64

// StreamingEngine is a steady-state (mu+lambda) GA its caller drives: it draws
// proposals, scores them out of band and returns each by CompleteEvaluation,
// every transition on its own goroutine under one mutex. The archive holds
// scored candidates alone, best first, so it is elitist by construction;
// unscored offspring wait in a bounded queue outside statistics and selection
type StreamingEngine[S Solution, F Numeric] struct {
	initializer InitializerFunc[S]
	selector    Selector[S, F]
	combiner    Combiner[S, F]
	perturbator Perturbator[S]
	cloner      Cloner[S]
	diversity   DiversityFunc[S, F]
	equal       EqualFunc[S]
	merge       MergeFunc[F]
	rank        RankFunc[S, F]
	due         DueFunc[S, F]
	schedule    ScheduleFunc[F]

	cfg StreamingConfig
	// rate and strength are the mutation knobs in force, the config's until a
	// schedule moves them
	rate, strength float64

	mu        sync.Mutex
	rngSource *rand.PCG
	rng       *rand.Rand
	archive   []Candidate[S, F]
	// seen is, per archive member, the id count when it was last scored or
	// handed out by Reevaluate
	seen []uint64
	// keys is the rank's scratch: a key a member and the entrant's, which is
	// why the archive has room for one over the pool
	keys []float64
	// version counts archive changes; the aggregates were computed at aggVersion
	version    uint64
	aggVersion uint64
	gen        int
	outcomes   int
	lastDiv    float64
	aggBest    F
	aggWorst   F
	aggAvg     F

	proposals ring[S]
	free      freeList[S]
	parents   []Candidate[S, F]
	offspring []S
	pending   pendingTable[S]

	nextID  uint64
	evicted uint64

	stats   atomic.Pointer[PoolStats[F]]
	started atomic.Bool
}

// NewStreamingEngine builds an engine. cloner may be nil, in which case solutions
// handed to callers alias engine memory and must be treated as read-only.
func NewStreamingEngine[S Solution, F Numeric](
	initializer InitializerFunc[S],
	selector Selector[S, F],
	combiner Combiner[S, F],
	perturbator Perturbator[S],
	cloner Cloner[S],
	config StreamingConfig,
) *StreamingEngine[S, F] {
	cfg := config.Normalize()
	rngSource := rand.NewPCG(cfg.Seed, cfg.Seed^defaultSeed)

	e := &StreamingEngine[S, F]{
		initializer: initializer,
		selector:    selector,
		combiner:    combiner,
		perturbator: perturbator,
		cloner:      cloner,
		cfg:         cfg,
		rate:        cfg.PerturbationRate,
		strength:    cfg.PerturbationStrength,
		rngSource:   rngSource,
		rng:         rand.New(rngSource),
		archive:     make([]Candidate[S, F], 0, cfg.PoolSize+1),
		seen:        make([]uint64, 0, cfg.PoolSize+1),
		keys:        make([]float64, cfg.PoolSize+1),
		parents:     make([]Candidate[S, F], 2),
		offspring:   make([]S, 2),
	}
	e.free.init(cfg.PoolSize)
	e.proposals.init(cfg.ProposalCapacity)
	e.pending.init(cfg.PendingCapacity)
	e.publishLocked()
	return e
}

// SetDiversity installs an optional diversity metric, recomputed once per generation
func (e *StreamingEngine[S, F]) SetDiversity(fn DiversityFunc[S, F]) {
	e.mu.Lock()
	e.diversity = fn
	e.mu.Unlock()
}

// SetEqual holds each genotype once, at its latest score: a repeat outcome
// moves the member to where its new score sorts, so an archive whose members
// are re-evaluated follows a changing objective. Without it an elitist archive
// fills with clones of its best member
func (e *StreamingEngine[S, F]) SetEqual(fn EqualFunc[S]) {
	e.mu.Lock()
	e.equal = fn
	e.mu.Unlock()
}

// SetMerge folds a repeat outcome into its genotype's score in place of
// replacing it, and counts it: MeanScore keeps a running mean, for a noisy
// objective that stays put. It acts only under SetEqual
func (e *StreamingEngine[S, F]) SetMerge(fn MergeFunc[F]) {
	e.mu.Lock()
	e.merge = fn
	e.mu.Unlock()
}

// SetRank keeps the archive by the keys fn gives its members in place of their
// scores: fn is asked on every outcome, of the archive whole with the entrant,
// and a full archive's last then leaves, the entrant where no key is under its
// own. So a key may weigh a member's Count, or its fellows, and a member many
// outcomes have tested need not yield to a newcomer's one
func (e *StreamingEngine[S, F]) SetRank(fn RankFunc[S, F]) {
	e.mu.Lock()
	e.rank = fn
	e.mu.Unlock()
}

// SetDue has Reevaluate hand out the member fn names in place of the one scored
// longest ago, for a host whose repeats follow what its outcomes prove
func (e *StreamingEngine[S, F]) SetDue(fn DueFunc[S, F]) {
	e.mu.Lock()
	e.due = fn
	e.mu.Unlock()
}

// MeanScore is the MergeFunc of a running mean, every outcome weighing alike
func MeanScore[F Numeric](old, new F, count int) F {
	return F(float64(old) + (float64(new)-float64(old))/float64(count+1))
}

// SetSchedule installs a schedule, consulted as each generation ends for the
// rate and strength later offspring are bred with; the configuration's hold
// until the first one does
func (e *StreamingEngine[S, F]) SetSchedule(fn ScheduleFunc[F]) {
	e.mu.Lock()
	e.schedule = fn
	e.mu.Unlock()
}

// Start enables proposal issuance and primes the queue. Idempotent and restartable
func (e *StreamingEngine[S, F]) Start() {
	if !e.started.CompareAndSwap(false, true) {
		return
	}
	e.mu.Lock()
	e.fillLocked()
	e.publishLocked()
	e.mu.Unlock()
}

// Stop halts proposal issuance; the archive is retained
func (e *StreamingEngine[S, F]) Stop() { e.started.Store(false) }

func (e *StreamingEngine[S, F]) Running() bool { return e.started.Load() }

// Propose returns the next unevaluated genotype and its evaluation id.
// The returned value is owned by the caller. A zero id means the engine is stopped
func (e *StreamingEngine[S, F]) Propose() (S, EvalID) {
	var zero S
	if !e.started.Load() {
		return zero, 0
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	sol, ok := e.proposals.pop()
	if !ok {
		e.produceLocked()
		if sol, ok = e.proposals.pop(); !ok {
			return zero, 0
		}
	}

	// Caller receives a recycled buffer; the engine retains the ring slice
	out := sol
	if e.cloner != nil {
		out = e.cloner.Clone(e.takeFreeLocked(), sol)
	}
	id := e.beginLocked(sol)
	e.publishLocked()
	return out, id
}

// BeginEvaluation registers an externally built genotype. The engine keeps a copy
func (e *StreamingEngine[S, F]) BeginEvaluation(sol S) EvalID {
	e.mu.Lock()
	defer e.mu.Unlock()

	kept := sol
	if e.cloner != nil {
		kept = e.cloner.Clone(e.takeFreeLocked(), sol)
	}
	id := e.beginLocked(kept)
	e.publishLocked()
	return id
}

// Reevaluate hands out the archive member due another evaluation, with its id:
// the one SetDue's hook names, or with none the one scored or handed out longest
// ago, the better of two that tie. A zero id means the engine is stopped, holds
// nothing, has no SetEqual, without which a repeat would enter as a copy, or
// its hook named no member
func (e *StreamingEngine[S, F]) Reevaluate() (S, EvalID) {
	var zero S
	if !e.started.Load() {
		return zero, 0
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.equal == nil || len(e.archive) == 0 {
		return zero, 0
	}
	due := 0
	if e.due != nil {
		if due = e.due(e.archive); due < 0 || due >= len(e.archive) {
			return zero, 0
		}
	} else {
		for i := 1; i < len(e.seen); i++ {
			if e.seen[i] < e.seen[due] {
				due = i
			}
		}
	}

	sol := e.archive[due].Data
	out, kept := sol, sol
	if e.cloner != nil {
		out = e.cloner.Clone(e.takeFreeLocked(), sol)
		kept = e.cloner.Clone(e.takeFreeLocked(), sol)
	}
	id := e.beginLocked(kept)
	e.seen[due] = uint64(id)
	e.publishLocked()
	return out, id
}

// CompleteEvaluation admits a scored genotype into the archive
func (e *StreamingEngine[S, F]) CompleteEvaluation(id EvalID, score F) {
	e.mu.Lock()
	defer e.mu.Unlock()

	sol, ok := e.pending.take(id)
	if !ok {
		return
	}
	e.insertLocked(Candidate[S, F]{Data: sol, Score: score, Count: 1})

	e.outcomes++
	if e.outcomes >= e.cfg.MinOutcomesPerGen {
		e.outcomes = 0
		e.gen++
		if e.diversity != nil {
			e.lastDiv = e.diversity(e.archive)
		}
		if e.schedule != nil {
			e.rate, e.strength = e.schedule(e.gen, e.statsLocked())
		}
		e.fillLocked()
	}
	e.publishLocked()
}

// AbandonEvaluation discards an in-flight evaluation without scoring it
func (e *StreamingEngine[S, F]) AbandonEvaluation(id EvalID) {
	e.mu.Lock()
	if sol, ok := e.pending.take(id); ok {
		e.recycleLocked(sol)
	}
	e.publishLocked()
	e.mu.Unlock()
}

// Reset drops in-flight evaluations and queued proposals; archive and generation persist
func (e *StreamingEngine[S, F]) Reset() {
	e.mu.Lock()
	e.pending.clear()
	e.proposals.clear()
	e.free.clear()
	e.outcomes = 0
	if e.started.Load() {
		e.fillLocked()
	}
	e.publishLocked()
	e.mu.Unlock()
}

// Stats returns the latest published snapshot without locking
func (e *StreamingEngine[S, F]) Stats() PoolStats[F] { return *e.stats.Load() }

func (e *StreamingEngine[S, F]) Generation() int   { return e.stats.Load().Generation }
func (e *StreamingEngine[S, F]) PendingCount() int { return e.stats.Load().Pending }
func (e *StreamingEngine[S, F]) EvaluationsStarted() uint64 {
	return e.stats.Load().Evaluations
}

// Best returns a copy of the top archive member
func (e *StreamingEngine[S, F]) Best() (Candidate[S, F], bool) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(e.archive) == 0 {
		return Candidate[S, F]{}, false
	}
	c := e.archive[0]
	if e.cloner != nil {
		var zero S
		c.Data = e.cloner.Clone(zero, c.Data)
	}
	return c, true
}

// Snapshot returns a deep copy of the archive for persistence
func (e *StreamingEngine[S, F]) Snapshot() *Pool[S, F] {
	e.mu.Lock()
	defer e.mu.Unlock()

	p := &Pool[S, F]{
		Members:    make([]Candidate[S, F], len(e.archive)),
		Generation: e.gen,
		Stats:      *e.stats.Load(),
	}
	for i, c := range e.archive {
		if e.cloner != nil {
			var zero S
			c.Data = e.cloner.Clone(zero, c.Data)
		}
		p.Members[i] = c
	}
	return p
}

// Inject replaces the archive with persisted candidates and takes ownership of
// them. Queued proposals derived from the old archive are discarded
func (e *StreamingEngine[S, F]) Inject(candidates []Candidate[S, F], generation int) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for i := range e.archive {
		e.recycleLocked(e.archive[i].Data)
	}
	e.archive, e.seen = e.archive[:0], e.seen[:0]
	e.version++

	for _, c := range candidates {
		c.Count = max(c.Count, 1)
		e.insertLocked(c)
	}
	e.gen = generation

	e.proposals.clear()
	if e.started.Load() {
		e.fillLocked()
	}
	e.publishLocked()
}

// --- internals (mu held) ---

func (e *StreamingEngine[S, F]) beginLocked(sol S) EvalID {
	e.nextID++
	id := EvalID(e.nextID)
	if old, hit := e.pending.put(id, sol); hit {
		e.evicted++
		e.recycleLocked(old)
	}
	return id
}

// insertLocked places a scored candidate into the descending archive, evicting
// the tail when full. Under SetEqual a genotype already archived leaves first
// and re-enters, at the score SetMerge makes of the two, so the archive never
// holds one twice. Under SetRank the keys order it and name who leaves
func (e *StreamingEngine[S, F]) insertLocked(c Candidate[S, F]) {
	n := len(e.archive)
	if e.equal != nil {
		for i := range e.archive {
			if !e.equal(e.archive[i].Data, c.Data) {
				continue
			}
			if held := e.archive[i]; e.merge != nil {
				c.Score, c.Count = e.merge(held.Score, c.Score, held.Count), held.Count+1
			}
			e.recycleLocked(e.archive[i].Data)
			copy(e.archive[i:], e.archive[i+1:n])
			copy(e.seen[i:], e.seen[i+1:n])
			n--
			e.version++
			break
		}
	}
	if e.rank != nil {
		e.rankLocked(c, n)
		return
	}
	if n == e.cfg.PoolSize && c.Score <= e.archive[n-1].Score {
		e.recycleLocked(c.Data)
		return
	}

	idx := n
	for idx > 0 && e.archive[idx-1].Score < c.Score {
		idx--
	}
	if n == e.cfg.PoolSize {
		e.recycleLocked(e.archive[n-1].Data)
		n--
	}
	e.archive, e.seen = e.archive[:n+1], e.seen[:n+1]
	copy(e.archive[idx+1:], e.archive[idx:n])
	copy(e.seen[idx+1:], e.seen[idx:n])
	e.archive[idx], e.seen[idx] = c, e.nextID
	e.version++
}

// rankLocked places c after the archive's first n and orders them all by the
// keys the rank gives, equal keys as they stood with c last, as a score orders
// them; the last of more than the pool holds leaves
func (e *StreamingEngine[S, F]) rankLocked(c Candidate[S, F], n int) {
	e.archive, e.seen = append(e.archive[:n], c), append(e.seen[:n], e.nextID)
	keys := e.keys[:n+1]
	e.rank(e.archive, keys)
	for i := 1; i <= n; i++ {
		for j := i; j > 0 && keys[j-1] < keys[j]; j-- {
			e.archive[j-1], e.archive[j] = e.archive[j], e.archive[j-1]
			e.seen[j-1], e.seen[j] = e.seen[j], e.seen[j-1]
			keys[j-1], keys[j] = keys[j], keys[j-1]
		}
	}
	if n == e.cfg.PoolSize {
		e.recycleLocked(e.archive[n].Data)
		e.archive, e.seen = e.archive[:n], e.seen[:n]
	}
	e.version++
}

// fillLocked tops up the proposal queue within the tick budget
func (e *StreamingEngine[S, F]) fillLocked() {
	if e.proposals.free() == 0 {
		return
	}
	if e.cfg.RefillMode == RefillDeterministic {
		for e.proposals.free() > 0 && e.produceLocked() {
		}
		return
	}
	deadline := time.Now().Add(e.cfg.TickBudget)
	for i := 0; e.proposals.free() > 0; i++ {
		if !e.produceLocked() {
			return
		}
		if i&3 == 3 && time.Now().After(deadline) {
			return
		}
	}
}

// produceLocked queues one offspring batch, or a random genotype while the
// archive is too small for selection. Reports whether anything was queued
func (e *StreamingEngine[S, F]) produceLocked() bool {
	if len(e.archive) < 2 {
		return e.proposals.push(e.initializer(e.rng))
	}

	e.selector.Select(e.archive, e.parents, e.version, e.rng)
	n := e.combiner.Combine(e.parents, e.offspring, e.rng)

	pushed := false
	for j := range n {
		e.perturbator.Perturb(&e.offspring[j], e.rate, e.strength, e.rng)
		if !e.proposals.push(e.offspring[j]) {
			return pushed
		}
		pushed = true
		e.offspring[j] = e.takeFreeLocked()
	}
	return pushed
}

func (e *StreamingEngine[S, F]) publishLocked() {
	s := e.statsLocked()
	e.stats.Store(&s)
}

// statsLocked is the engine as it stands, its aggregates redone if the archive moved
func (e *StreamingEngine[S, F]) statsLocked() PoolStats[F] {
	if e.aggVersion != e.version {
		e.aggVersion = e.version
		e.aggBest, e.aggWorst, e.aggAvg = 0, 0, 0
		if n := len(e.archive); n > 0 {
			var total F
			for i := range e.archive {
				total += e.archive[i].Score
			}
			e.aggBest, e.aggWorst, e.aggAvg = e.archive[0].Score, e.archive[n-1].Score, total/F(n)
		}
	}
	return PoolStats[F]{
		BestScore:    e.aggBest,
		WorstScore:   e.aggWorst,
		AverageScore: e.aggAvg,
		Diversity:    e.lastDiv,
		Size:         len(e.archive),
		Generation:   e.gen,
		Pending:      e.pending.live,
		Evaluations:  e.nextID,
		Evicted:      e.evicted,
	}
}

// recycleLocked keeps a buffer for later offspring; without a Cloner the
// buffer may be what a caller holds, so nothing is kept
func (e *StreamingEngine[S, F]) recycleLocked(sol S) {
	if e.cloner != nil {
		e.free.put(sol)
	}
}

func (e *StreamingEngine[S, F]) takeFreeLocked() S { return e.free.take() }
