# GADAR process casebook: how ideas were tried, diagnosed, and rejected

The evidence base for `../SKILL.md`. The skill states the rules; this file
holds the real episodes they were compressed from, so every rule stays
checkable against what actually happened. When a new episode teaches a new
lesson (or contradicts a rule), it gets added here in the same commit that
changes the skill — rules without episodes are opinions.

Sections: Part 1 = five failure patterns (F1–F5) and the question that
resolved each; Part 2 = the rejection taxonomy with epistemic statuses;
Part 3 = the pre-registration habit; Part 4 = the question checklist;
Part 5 = a worked example of applying the process to ideas transferred
into a sibling project (GammaZero), useful as a template for future
streams in THIS repo inheriting GADAR's conclusions.

---

## Part 1 — When an idea failed: five failure patterns and the question
that resolved each

### F1. Failure with pre-named diagnosis lenses (don't debug blind)

The plan's Phase-2 gate came with its failure branch WRITTEN IN ADVANCE:
"if zero-shot does not beat the control, debug with the aliasing /
expressiveness lenses before adding machinery." When structural
featurization underperformed and produced type-invalid grounded actions
(4-argument gripper groundings), the pre-named lenses turned a vague
failure into a differential diagnosis: is this aliasing (two domains
structurally identical, semantically different — unfixable by
construction), expressiveness (the GNN cannot compute what it needs — needs
architecture), or missing information (the graph never contained the
fact — needs representation)? The answer was the third: two typing
conventions existed in the suite (declared types vs unary-predicate types),
and the featurizer could only see one. That diagnosis produced type
COMPILATION — normalize both conventions into one at the spec level — not
"add a type feature channel" and not "filter invalid actions after
decoding," both of which would have patched the symptom.

**Question to institutionalize:** before the experiment runs, write down
the 2–3 named ways it could fail and what each failure would look like in
the output. A failure you can classify is a step; a failure you can only
observe is a stall. And when diagnosing: is this symptom a missing FACT, a
missing CAPABILITY, or a structural impossibility? The fixes live in
different places (representation / architecture / claim scope) and patching
the wrong layer wastes a cycle.

### F2. Plateau: ask what the system doesn't KNOW, not what it doesn't have

Binding (BIND rung) clearly beat the union control zero-shot, then
plateaued far below hopes on the harder domains. The reflex answers — more
message-passing rounds, bigger model, more data, more epochs — were all
available and all wrong. The question that worked: what does the network
know about an action? Answer: everything about what it NEEDS (binding =
preconditions in lifted vocabulary), nothing about what it LEADS TO
relative to the goal. That gap analysis produced the chain layer
(effect→precondition edges + goal-distance features), which became the
headline component. It was not in the plan; it came from interrogating the
plateau.

**Question:** when performance plateaus, enumerate what information the
model provably has access to versus what a competent human planner would
use in the same state. The difference is a representation to-do list;
capacity knobs come only after that list is empty.

### F3. A good idea that measures ~zero: ask what MASKS it before discarding

Batched evaluation (one model call per rollout round across all problems)
was obviously right and measured at 0.5% improvement — essentially noise.
The idea was not discarded; the question became "what regime is this test
in?" A profile showed a serial, kernel-bound cost (dense edge-tensor
allocation) dominating everything: batching the model call cannot help
while 90% of the time is spent zero-filling pages. After the sparse-edge
fix removed the dominant term, the SAME batching was worth 1.25x, and later
became the substrate for the worker pool (2.5x more).

**Question:** a null result for a mechanistically sound idea is a claim
about the test regime, not about the idea. Before discarding: what is the
current dominant cost/error term, and could it be hiding the effect? Fix
the dominant term, re-test the idea, THEN judge it.

### F4. Contradictory evidence: calibrate the instrument, then move the
hypothesis (two episodes)

(a) The batched and sequential harnesses reported different coverage (70%
vs 80%, same model). First hypothesis: a bug in the new batched code. The
author pushed back — extensive prior testing said both harnesses were
sound. Taking that evidence seriously reframed the question from "which
harness is broken" to "what could make two CORRECT harnesses differ?"
Answer: one shared RNG consumed problem-by-problem in one harness and
interleaved in the other — both correct, divergent by construction. The fix
(per-problem seeding) came from the reframed question; the original
bug-hunt would never have found it because there was no bug. A second
lesson hides inside: per-step action traces showed NO difference (the
fallback picks among already-scored proposals without a model call), so the
divergence was invisible to the existing instrumentation — which is why the
parity harness was extended to compare final OUTCOMES, not just traces.

(b) The byte-identity fingerprint tool, on its first real use, reported
avalanche mismatches — hundreds of differing tensors — after a refactor
believed safe. Before concluding the refactor was broken, the tool was run
against ITSELF (same code, two invocations): same avalanche. The
instrument, not the code, was nondeterministic (set iteration order follows
the interpreter's hash seed, and the rollouts diverged). The tool got a
deterministic sort and a mandatory self-consistency step; the refactor was
fine.

**Questions:** when your hypothesis says "bug" and the evidence says "both
sides tested clean," construct the scenario in which BOTH are right — it
usually names a shared hidden dependency (RNG streams, iteration order,
global state). And never trust an alarming measurement from an instrument
that has not been run against itself.

### F5. A sound concept that failed once: cost-model the failure, redesign,
retry

Parallelizing graph construction was tried early in the project and came
out SLOWER — and was shelved with cause unknown. When evaluation speed
became existential, it was revisited with a cost model instead of a
retry: measure what crosses the process boundary and what that costs.
Finding: shipping one pddlgym state to a worker costs ~4.6ms in
serialization against ~5.7ms of featurization work — a task-shipping pool
spends ~half its budget on the pipe. The redesign followed from the number:
workers OWN problems (state never crosses the boundary; only small numpy
results come back), envs are built inside the workers, and the pool now
delivers 2.5x. Same concept, opposite design, and the difference was one
measurement that could have been taken before the first attempt.

The same discipline then killed the next "obvious" extension before it
shipped: converting to framework tensors inside the workers looked like a
free 9% and measured at 65s→631s (framework pickler pathology). Measured,
reverted, recorded with the number.

**Question:** for any parallel/distributed idea, measure serialization
cost against unit-of-work cost FIRST — the answer dictates the design
(ship tasks vs own state), and it costs one benchmark. More generally: a
failed attempt at a sound concept is a design datum, not a verdict; extract
the cost model from the failure before shelving the concept.

---

## Part 2 — When an idea was rejected WITHOUT being tried: the taxonomy,
with the epistemic status each rejection deserves

GADAR rejected many ideas that might have worked. What kept this honest was
that each rejection has a BASIS and a recorded epistemic status — and the
statuses are different in kind. Mislabeling one as another corrupts future
planning (yours or a sibling project's).

| Basis | GADAR example | Status label | What it licenses later |
|---|---|---|---|
| **Refuted by decisive measurement** | stateless worker pool (pickle ≈ work); PyG-in-workers (65→631s); threading (GIL-bound); repr caching (already cached upstream); oversubscription (measured harmful) | *refuted-here* | Do not retry in this codebase; RE-ASK in a different stack — the GIL argument is Python-specific, the serialization ratio is workload-specific |
| **Subsumed by a chosen alternative** | Method A (compiled): the joint path's lifted symbol nodes already provide the global-hub role, and one-featurizer-one-pipeline made A a second codebase | *subsumed, unattempted* | The idea is NOT refuted. Any doc citing it must say "no evidence." It can be revived when the subsumption argument breaks (in GammaZero it does: MCTS cost inverts the affordability argument) |
| **Different-not-better (no implicating failure)** | GOOSE-style random-unit-vector position features vs the incumbent capped one-hot positions: the comparison was planned, and never run — because position encoding was never implicated in ANY observed failure | *unexamined, unimplicated* | Ideas earn experiments by being implicated in failures. If a failure later points at positions, the comparison is back on the table |
| **Excluded to protect claim semantics** | PDDLFuse generated domains as training augmentation: might have improved zero-shot numbers, would have muddied what "zero-shot on unseen domains" means | *excluded-by-design* | Revisit only alongside an explicit claim change; numbers and their meaning move together |
| **Deferred by risk sequencing** | Lifted decoding (constructing actions without pre-enumeration): declared future work on day one — "start grounded, parity with the working system, change one thing at a time" | *deferred* | A standing invitation, gated on the current layer being solid |
| **Outsourced to external evidence** | The cross-domain value-regression ablation (C3): theory plus two independent published results (GOOSE-DI's collapse, Müller et al.'s ranking-margin fix) already answered it; the experiment's marginal information did not justify its cost when compute got tight | *citation-carried* | Honest only if labeled: the paper cites, it does not claim to have run it |
| **Mechanism-checked and declined** | "More attention heads?" — read the code first: heads in this architecture are averaged with a shared scoring projection — an ensemble, not capacity. "More epochs?" — the loss curve was flat from halfway. Both declined without a run | *mechanism-precluded* | The check costs minutes and beats a GPU-day; and "a large effect from this knob would be a reason to re-read the code, not to celebrate" |
| **Budget triage with explicit cut order** | Ablation splits trimmed 8→4 (with the 4 chosen for named coverage: canonical / family-transfer / structurally-distinct / largest-vocabulary); diversity curve trimmed; a written "cut in this order" list ending in "never cut: the floor, the control, the binding ablation — without these there is no paper" | *cut-ranked* | Cuts ordered by claims served, decided ONCE in calm, not repeatedly under deadline |

Two meta-rules over the whole table:

1. **Unattempted ≠ refuted.** GADAR's docs mark Method A "not refuted,
   unattempted" precisely so a sibling project doesn't inherit a
   non-existent negative result. Every rejection you record must carry its
   status, because the status — not the rejection — is what transfers.
2. **Every rejection is written down WITH its basis** (the
   rejected-approaches sections in the performance docs, the standing
   decisions in the north star). The next person will have the same
   good-looking idea; the record either saves them a week or hands them
   the exact measurement to re-take under new conditions.

---

## Part 3 — The pre-registration habit (measure-either-way beats
measure-if-worried)

The plan pre-registered measurements whose outcome was uncertain and
declared both outcomes informative:

- The "expressiveness tax" (does deleting identity features hurt
  training-domain performance?) was ordered measured "regardless." It came
  out REVERSED — structure beat identity even in-domain (34.8 → 60.5
  combined) — and became one of the paper's most useful findings. Had it
  been measured only "if things look bad," the reversal would never have
  been seen.
- C4-style questions were framed "informative either way" in advance
  (joint wins → meaning is contextual; tie → conditioning is amortizable),
  so no outcome could be a disappointment, only an answer.
- The union control's PREDICTED failure (near-zero zero-shot) was itself
  the experiment: the prediction was written down first, so the near-zero
  result certified the control instead of looking like a broken run.

And the infrastructure version of the same habit: floors and controls were
built BEFORE the systems they calibrate (random-policy floor, union
control), because a number without its floor is uninterpretable, and a
floor built after the fact invites motivated choices.

**One counter-example, recorded as a fault:** checkpoint selection
infrastructure embedded the assumption "best loss = best model." Zero-shot
transfer violates it (transfer peaks well before training loss bottoms out)
and the loss-ranked slots were silently discarding the best-transferring
checkpoints until periodic snapshots were added. The assumption was never
pre-registered, so its violation was discovered late and by accident.
When the objective changes (in-domain → transfer; returns → guidance
quality), re-derive every selection rule from scratch.

---

## Part 4 — The question checklist for every new idea

Distilled from the episodes above. GADAR did not follow this list because
it existed; the list exists because following its items is what worked and
skipping them is what cost time.

Before building:
1. **Which claim does this serve?** No claim, no code — however elegant.
2. **What observed failure implicates it?** Ideas earn experiments by
   being implicated. An idea addressing no observed failure goes to the
   *unexamined* shelf, not the build queue.
3. **What is the mechanism by which it should win?** Read the code /
   architecture first; some knobs cannot deliver the hoped effect
   (heads-as-ensemble). Minutes of reading vs days of compute.
4. **Is it subsumed** by something already chosen? If claimed so, write
   down the subsumption argument — it is an assumption that can break in a
   new setting, and the record is what lets a successor notice.
5. **What is the cheapest decisive measurement?** Often one benchmark
   kills or licenses the whole design (serialization-vs-work). Take it
   before designing, not after failing.
6. **What does it cost in code paths?** A second pipeline is a standing
   tax on every future change; GADAR's one-featurizer discipline is why
   its ablation ladder was runnable as configs. If the idea forks the
   pipeline, price that fork as part of the idea.
7. **Does it contaminate a claim?** (Training augmentation vs zero-shot
   semantics.) Numbers and their meaning move together.

While testing:
8. **Pre-register the measurement and both interpretations.** Declare
   what each outcome would mean before seeing it; measure "regardless,"
   not "if worried."
9. **Is the test regime capable of showing the effect?** Know the current
   dominant term; a null result under a dominating confound is not a
   verdict (batching at 0.5%).
10. **Is the instrument calibrated?** Run comparison tools against
    themselves; build outcome-level checks, not just step-level ones,
    before trusting agreement.

After the result:
11. **Classify the failure** (missing fact / missing capability /
    structural impossibility) before choosing where to fix it.
12. **On surprise, construct the scenario where both the evidence and the
    prior are right** before declaring either wrong (the two-correct-
    harnesses episode).
13. **Record the outcome with its epistemic status** — refuted-here /
    subsumed-unattempted / unexamined / excluded-by-design / deferred /
    citation-carried / mechanism-precluded / cut-ranked — and the
    measurement or argument that produced it.
14. **Negative results go in the log, not the chat.** They are data with
    a shelf life longer than the conversation that produced them.

---

## Part 5 — Applying this to the GADAR-derived ideas now entering GammaZero

The transferred ideas arrive with the SAME status GADAR's original plan
had: plausible, mechanism-backed, untested in their target setting. Expect
first-contact failures, and expect them to be informative only if the
scaffolding exists first. Concretely:

- The **chain-layer analog** (sense→affect→reward chains) may plateau or
  even hurt: POMDP "leads to" runs through belief updates, not set
  arithmetic, and the relaxed-reachability sketch that was cheap in PDDL
  may be a poor approximation under uncertainty. If it fails, apply F2:
  enumerate what the network provably has vs what an information-gathering
  planner uses — the gap analysis in belief space may name a different
  missing structure than chaining (e.g. expected-information-flow rather
  than reachability). That would be a finding, not a failure.
- The **one-builder-two-modes** design for compiled-vs-joint is itself a
  hypothesis about YOUR codebase's structure. If the RIR cannot host both
  as flags, that discovery should be made in a design spike (cheapest
  decisive test, item 5), not three weeks into building mode two.
- The **identity-features reversal** may not replicate: GammaZero deletes
  near-policy semantic features, not symbol one-hots (the report flags
  this). Pre-register the tax measurement either way (item 8) — a reversal
  or a real tax are both publishable calibrations of the prober's job.
- Every rejection GADAR recorded as *refuted-here* is up for RE-ASKING in
  Julia (item: the GIL argument doesn't exist there; serialization ratios
  differ under Distributed). The statuses transfer; the verdicts do not.

The one-sentence version: **GADAR's outcomes are hypotheses in GammaZero's
setting; GADAR's process is not.** Follow the checklist, keep the epistemic
bookkeeping, and a first-contact failure becomes the next design input —
which is exactly what happened, twice, on the way to the system that
worked.
