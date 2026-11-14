---
name: research
description: >-
  Research-methodology discipline for this repo, distilled from the GADAR
  project (2026). Use whenever experimental results arrive and need
  interpreting (a run finished, a number is surprising, an ablation failed,
  coverage dropped, two harnesses disagree); whenever choosing which idea or
  experiment to try next under limited GPU time; before building any new
  representation, ablation, or pipeline change; when starting a new research
  stream or writing its north-star document; and when recording a rejected
  idea or negative result. Also use when the user asks "what should I try
  next", "why didn't this work", "is this result real", or wants to plan
  experiments — even if they don't mention the word research.
---

# Research process

This skill encodes how research was actually practiced in this repo during
the GADAR project — the questions asked when results arrived, and the basis
on which ideas were selected or rejected under finite time. It exists
because the researcher who built that expertise may return rusty, and
because future agents in this repo should inherit the process, not just the
code. Every rule here is anchored to a real episode in
`references/casebook.md` (referenced as F1–F5 and the rejection table);
read the referenced episode whenever a rule's rationale isn't obvious —
the episodes are the evidence, the rules are just the compression.

First identify the situation, then follow that section:

| Situation | Section |
|---|---|
| A result just arrived and needs interpreting | §1 |
| Choosing what to try next (many ideas, little time) | §2 |
| About to build something new | §3 |
| Starting a new research stream in this repo | §4 |
| Recording a decision, rejection, or negative result | §5 |

## §1 Reading a result

**Step 0 — before believing anything: is the instrument calibrated?**
A comparison tool must be run against itself (same code, two invocations)
before its first real use; nondeterministic iteration order produces
avalanche false alarms indistinguishable from real breakage (casebook
F4b). Agreement checks must cover final outcomes, not only per-step
traces — a shared RNG consumed in different orders changed reported
coverage 10 points with byte-identical traces (F4a). If the instrument
hasn't been calibrated, do that before interpreting the result.

Then classify the result and ask the matching question:

- **Confirming, and pre-registered** → the most trustworthy kind of
  finding. Record it with numbers; if it was pre-registered with both
  interpretations written down, the interpretation is already done — use
  it, don't re-litigate.
- **Null result for a mechanistically sound idea** → this is a claim about
  the test regime, not the idea. Ask: what is the current DOMINANT
  cost/error term, and could it mask the effect? Profile, fix the dominant
  term, re-test, then judge. (Batching measured 0.5% under a kernel-bound
  serial cost; after removing it, the same batching gave 1.25x and became
  the substrate for a further 2.5x — F3.)
- **Failure** → classify before fixing: is the symptom a missing FACT
  (representation), a missing CAPABILITY (architecture), or a structural
  IMPOSSIBILITY (claim scope)? The fixes live in different layers and
  patching the wrong one wastes a cycle. Use the failure lenses that were
  named in advance for this experiment (§3 requires naming them); a
  failure you can classify is a step, one you can only observe is a stall.
  (Type-invalid groundings were a missing-fact failure; the fix was
  compiling types at the spec level, not filtering symptoms — F1.)
- **Plateau** → ask what the system doesn't KNOW, not what it doesn't
  have. Enumerate what information the model provably has access to versus
  what a competent solver of the same problem would use; the difference is
  a representation to-do list. Capacity knobs (width, depth, epochs, data)
  come only after that list is empty. (This question, not a brainstorm,
  produced GADAR's headline component — F2.)
- **Contradiction with prior evidence** → before declaring either side
  wrong, construct the scenario in which BOTH are right; it usually names
  a shared hidden dependency — RNG streams, iteration order, global state,
  a selection rule (F4a). Take the other party's evidence seriously
  especially when it contradicts your bug theory.
- **Surprise** → check the instrument (step 0), then check whether the
  measurement was pre-registered. A surprise in a pre-registered
  measurement is a finding (the identity-features reversal was one); a
  surprise in an ad-hoc measurement is a hypothesis needing a designed
  test.

Whatever the class: numbers go in the results log/ledger with hardware
labels, never only in the conversation. Timings are not comparable across
hosts.

## §2 Selecting the next idea

Ideas are infinite; GPU-days are not. Run every candidate through these
questions IN ORDER — most ideas die cheaply at questions 1–3, which is the
point:

1. **Which claim does it serve?** No claim, no code, however elegant. If
   it serves no claim in the north star, it is scope creep or a new claim
   — decide which, explicitly.
2. **What observed failure implicates it?** Ideas EARN experiments by
   being implicated in failures. An idea addressing no observed failure
   goes on the *unexamined* shelf (§5), not the build queue. (A planned
   position-encoding comparison was never run because positions never
   appeared in any failure diagnosis — and that was correct triage.)
3. **What is the mechanism by which it should win?** Read the code first;
   some knobs cannot deliver the hoped effect (attention heads that
   average with a shared projection are an ensemble, not capacity).
   Minutes of reading beat days of compute — and a large effect from a
   mechanism-precluded knob is a reason to re-read the code, not
   celebrate.
4. **Is it subsumed by something already chosen?** If so, write the
   subsumption argument down — it is an assumption that can break in a new
   setting, and the record lets a successor notice when it does.
5. **What is the cheapest decisive measurement?** Often one benchmark
   kills or licenses the whole design before it is built (a serialization-
   vs-work measurement dictated the entire worker-pool design — F5). Take
   it before designing, not after failing.
6. **What does it cost in code paths?** A second pipeline is a standing
   tax on every future change. Prefer new capability as a MODE over the
   existing path (a config, a featurization flag) rather than a fork —
   this repo's ablation ladder was runnable as configs precisely because
   of one-featurizer discipline. If the idea requires a fork, price the
   fork as part of the idea.
7. **Does it contaminate a claim?** Some ideas improve numbers while
   weakening what the numbers mean (training augmentation vs a zero-shot
   claim). Numbers and their meaning move together.

Ranking among survivors: existential claims before win-either-way
analyses; failure-implicated before speculative; cheapest-decisive-test
first among ties. When the schedule is at risk, write the cut order ONCE,
in calm — ranked by claims served, ending with a "never cut" list — and
then follow it, instead of re-deciding repeatedly under deadline.

## §3 Before building

- **Pre-register the measurement and both interpretations.** Write down
  what each outcome would mean BEFORE seeing it, and measure "regardless",
  not "if worried" — the measure-regardless items produced the most
  valuable surprises (the feared expressiveness tax measured as a
  reversal). Frame two-sided questions so either answer is an answer.
- **Name the failure lenses in advance.** For each experiment, write the
  2–3 ways it could fail and what each looks like in the output (§1's
  failure classification depends on this).
- **Gates before experiments.** Each phase gets a go/no-go gate with its
  criterion written before the phase runs. The baseline-parity gate is
  permanent: never break reproduction of the prior system through the new
  harness.
- **Floors and controls before systems.** A number without its floor is
  uninterpretable, and a floor built afterward invites motivated choices.
  Build the random/untrained floor and the control baseline first; the
  control's PREDICTED failure, written down in advance, is itself a
  certification that the control works.
- **Selection rules are hypotheses.** Any "keep the best X" rule embeds an
  assumption about what best means; when the objective changes (in-domain
  → transfer), re-derive the rule — best-loss selection silently discarded
  the best-transferring checkpoints here until periodic snapshots were
  added.
- **Verification apparatus scales with what it protects.** Refactors of
  hot paths get byte-identity gates (fingerprint before/after); harness
  changes get outcome-level parity across all execution modes; anything
  stochastic gets per-unit (per-problem/per-episode) seeding BEFORE any
  batching or interleaving exists, because retrofitting it is how the
  70%-vs-80% coverage bug happened.

## §4 Starting a new research stream

New streams in this repo follow the north-star pattern that worked (see
CLAUDE.md as the exemplar — its §2/§3/§6/§8 are the template):

1. **The contribution stated once**, with its category (capability vs
   insight) and the one-line version for intros.
2. **A claims table**: every experiment certifies exactly one claim, every
   claim has exactly one certifying experiment. Mark each claim's risk:
   existential (the stream dies if it fails) vs informative-either-way.
3. **Positioning verified against primary sources, read in full** — with
   the specific facts to cite correctly written down, so writing time
   doesn't re-derive them. Never claim "first X" without the rubric table
   that shows it.
4. **Phases with go/no-go gates** and pre-named failure lenses per gate.
   Later phases are intentions, not commitments — re-plan at each gate.
5. **Rules for code**, including: every change names its claim; the
   baseline is sacred; invariance is tested not assumed; negative results
   are recorded; extend rather than rebuild; keep the map current.
6. **Open decisions listed with the phase that forces each** — decided
   then, not before, and recorded when decided.

Streams sharing this repo share its infrastructure discipline: one
featurizer/harness with modes, caches keyed and versioned, checkpoint
identity including everything that changes shapes or meaning.

## §5 Recording decisions (epistemic bookkeeping)

Every rejection, negative result, and standing decision is recorded WITH
its basis and epistemic status — the status, not the rejection, is what
transfers to future streams and sibling projects:

| Status | Meaning | What it licenses later |
|---|---|---|
| *refuted-here* | killed by a decisive measurement in this stack | don't retry here; RE-ASK under different conditions (different language, workload, scale) |
| *subsumed-unattempted* | dropped because an alternative covers it | NOT refuted — revivable the moment the subsumption argument breaks |
| *unexamined* | no observed failure implicates it | back on the table the moment a failure points at it |
| *excluded-by-design* | would contaminate a claim's meaning | revisit only alongside an explicit claim change |
| *deferred* | right idea, wrong layer of risk right now | a standing invitation, gated on the current layer being solid |
| *citation-carried* | answered by external evidence, not run here | honest only if labeled as cited, never as run |
| *mechanism-precluded* | the code can't deliver the hoped effect | re-check after architecture changes |
| *cut-ranked* | budget triage per the written cut order | first to revive when resources return |

Two absolutes: **unattempted is never written as refuted**, and rejected
ideas are recorded with the measurement or argument that killed them —
the next person will have the same good-looking idea, and the record
either saves them a week or hands them the exact measurement to re-take.

## Maintaining this skill

This skill is a living document with the same contract as CLAUDE.md's map:
when an episode teaches a new lesson or shows one of these rules to be
wrong, update this file AND `references/casebook.md` (add the episode) in
the same commit. Rules without episodes are opinions; keep the evidence
attached.
