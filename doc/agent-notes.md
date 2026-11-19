# Agent Notes

What agents working on this repository learned that no rule in
[AGENTS.md](../AGENTS.md) and nothing in the [README](../README.md) states: facts
that cost a session an hour to rediscover, and how to check them. Read it before
starting. When a task teaches you one, add it in the same PR, one bullet under its
topic; delete a note that proves wrong or that the README now states, and link the
README instead.

## Genetic

- `StreamingStateVersion` names the layout `Checkpoint` writes and `Restore`
  reads. A change to `StreamingState`'s layout bumps it, and a host that carries
  the checkpoint in a capture of its own bumps that capture's version too: vif's
  capture holds `registry.SpeciesState`, which embeds it.
- A program over the library is tuned by its `-headless` table, several seeds
  at once in the background with a `sed` variant per constant, not by watching
  it: `examples/lander`'s crash rate at one setting varies by half between
  seeds, and its archive statistics read well long before its streams do.
- An elitist archive under a noisy objective selects for luck: settle its
  scores by repeats ([README](../README.md#repeat-outcomes)) before touching
  the operators.
- A noisy search over a network's weights was measured on a learner kept out of
  the tree, to rebuild before changing what it settled: 148 weights (16 inputs,
  six `tanh` units, a direct term) from a hand-written policy, one evaluation
  in flight, on a scrolling game scored by ticks survived and kills. Read the
  mean over 64 held-out rounds of the best member scored twice, twelve seeds a
  setting: one setting's seeds span 3,300 to 4,200. A second objective, the
  network fitted to a teacher's outputs through noisy batches, does not
  saturate, and shows a stall the game hides. Every row spends 12,000 rounds
  on the game and 36,000 batches on the fit, at a step of 0.03 of the range:

  ```
                                               game   fit, squared error
  one round an evaluation, no repeats          3070   19
  mean of three, no repeats                    3360    3.7
  one round, every third a Reevaluate          3330    1.5
  one round, the same under MeanScore          3545    1.6
  mean of three, every third, either way       3570    1.6
  separable CMA-ES, mean of three              3290    1.4
  ```

- On the same learner a step of 0.1 left one seed in twelve at its seed genome
  for 4,000 evaluations, and 0.02 to 0.05 none. `AdaptiveStrength` from 0.1
  ends level with the best fixed step at a `lo` of 0.02 (3610, 1.7) and
  stalls at 0.005 (3450, 15). A share of repeats from a half to a fifth, a
  pool of 16 or 32 and `BlockCombiner` against `UniformCombiner` all end
  within the seeds' spread of one another.
- `Reevaluate`'s order was tried two other ways there, a new member first and
  the fewest outcomes first: 3450 and 3370 against 3570 by age under a
  schedule, level without one. A schedule following the average score, and one
  decaying by generation, ended level with `AdaptiveStrength` at its floor.
- A rank is asked on an outcome and never between: a `SetRank` hook that reads
  its host's state, a running mean say, orders the archive as that state stood
  at the last outcome, which is the order `Snapshot` and a checkpoint show.
  Keyed by `Score` it must keep an unranked engine's checkpoint, which is how
  a host proves its hook is wired before it reads anything off it.
- A ranked archive was measured on vif's learner, whose rounds are that noisy
  game: members of eight outcomes or more held above the rest, both by a mean
  weighed with eight outcomes at the recent mean, a newcomer displacing only
  one of the rest. A member tested over fifty outcomes no longer leaves, and
  what a run fields held out is the unranked archive's within what luck moves.
  Entering on the first outcome, with no second to confirm it, fielded a
  quarter less.
- `Engine.Checkpoint`'s members alias the engine's buffers, which the next
  generation writes over: copy them before running the source engine on, as its
  test does. `Restore` into an engine of another configuration or pool size is
  refused rather than reshaped.
- An assignment genome under `CategoricalPerturbator` wants a rate of one over
  its length and a `MixProbability` near one, so most children differ from a
  parent by one move.
