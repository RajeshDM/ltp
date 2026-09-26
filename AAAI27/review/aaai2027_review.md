# GADAR — AAAI 2027 Review

*Organized by review form field. Each section below can be pasted directly into the corresponding box.*

---

## TITLE

Promising framing for domain-conditioned policies, but the headline claim rests on one domain and the ablation does not isolate the contribution it claims to isolate.

---

## PAPER SUMMARY

The paper addresses a structural limitation in learned policies for classical planning: existing policies (ASNets, relational-GNN value policies, GABAR) encode the domain vocabulary inside the network — as per-schema modules, per-predicate parameters, or vocabulary-sized feature dimensions — so a trained policy cannot even be *executed* on a domain with unseen predicates and schemas. Learned heuristics have already moved past this (STRIPS-HGN's structural features, GOOSE's lifted learning graph, which takes the domain description as input and is evaluated on held-out domains), but heuristics delegate action construction to search. The paper takes the corresponding step for policies, which Chen, Thiébaux, and Trevizan (2024) named as open.

The proposal is a policy π(s, G, D) that takes the PDDL domain description as an explicit input and constructs grounded actions in unseen domains zero-shot and without search. The encoder (GNN) and decoder (GRU, autoregressive schema-then-arguments ranking) are inherited unchanged from GABAR; the contribution is the graph they operate on. The representation places a *lifted domain layer* (schemas, predicate symbols, and one occurrence node per literal occurrence in preconditions and effects) alongside GABAR's ground instance layer, and connects them with two proposed relations: **binding edges**, from an object at a slot of an applicable ground action to the lifted precondition occurrences that slot satisfies, and **chain edges**, from an add-occurrence to a pre-occurrence of the same predicate (*enables*) and from a delete-occurrence likewise (*threatens*). All node and edge features are flags or one-hots whose widths depend only on global arity caps, so no dimension scales with |A|, |P|, |T|, or |O|, and renaming invariance holds by construction.

The evaluation is organized as a three-rung ladder — UNION (symbol one-hots over the union vocabulary; multi-task learning), GADAR-BIND (structural features, lifted layer, binding edges), GADAR (chain layer, grounded effect edges, goal-distance and goal-relevance features) — with a leave-one-domain-out protocol over eight standard benchmark domains and held-out results reported for Visitall, Miconic, and Grid on two splits each. The headline result is 68% zero-shot coverage at plan quality 0.97 on held-out Visitall's large test instances, against 4% without the chain layer, 0% for UNION, and 0% for an untrained control under an identical executor. A companion in-domain table reports a single eight-domain model at 60.5% combined coverage against 92.5% for eight per-domain GABAR models.

### Exact contribution, and whether the methodology and experiments can establish it

Stating the contribution precisely is necessary here, because the paper's claims sit at three different levels and the evaluation supports them unevenly.

- **C1 (capability claim).** A single learned policy that accepts the domain description as input and is *executable* — produces well-formed grounded actions — on a domain whose predicates and schemas were never seen in training, with no search and no per-domain component in the network.
- **C2 (representation claim).** Two specific novel relations, binding edges and chain edges, each computable in any domain, are what supply the evidence a policy (as opposed to a heuristic) needs, and each is individually responsible for a measured step in performance.
- **C3 (transfer claim).** The resulting policy does not merely execute but is *competent* zero-shot, at plan quality comparable to a satisficing planner.

**Does the methodology explain the contribution?** For C1, yes, and cleanly. The argument that every feature width is a function of the arity caps alone, that selection is always a dot product against a node present in the graph, and that the argument loop reads arity from D under a global cap, is the correct and complete explanation of why the network is executable in an unseen domain. Renaming invariance follows immediately and is verified as a unit test. This is the strongest part of the paper.

For C2, the methodology is *described* but not *established*. The justification for chain edges is an informal expressiveness argument — that the producer–consumer pairing survives only as a two-hop path through a predicate hub whose aggregation destroys it. The paper cites Chen, Thiébaux, and Trevizan's Theorem 4.3 for the analogous claim about LLG but proves nothing about its own graph. A formal statement — a pair of tasks whose ground-plus-lifted graphs are indistinguishable to L rounds of message passing without E_chain and distinguishable with it — is a small amount of work and would convert the paper's central intuition into a result. The same holds for the binding layer: the claim that it "supplies the evidence" a policy needs is asserted, never characterized.

**Do the experiments ask the right questions?** The high-level design is well judged. Leave-one-domain-out is stricter than the fixed split used in prior domain-independent evaluation and this is correctly noted; the UNION control isolates representation from architecture and data; the untrained control isolates the policy from the revisit monitor; and the achievement oracle probes whether the policy is only a terminal trigger. These are the right four questions, and posing them explicitly is commendable.

But the protocol does not deliver the isolation it claims. The paper asserts three times — in the introduction, in *A Ladder of Domain Conditioning*, and in *Ablations and oracle* — that "adjacent systems differ in exactly one component" and that "the GADAR-BIND-to-GADAR step isolates the chain layer." By the paper's own definition, it does not. That step adds **four** things at once: E_chain, E_eff (the grounded achieves-goal / deletes-true edges), the bucketed goal-distance one-hot, and the goal-relevance bit. The UNION-to-GADAR-BIND step similarly bundles three: removal of symbol identity, addition of the lifted layer (Σ, Q, E_inst, E_occ), and addition of E_bind. Consequently **neither of the two claimed novel relations is isolated by any experiment in the paper**, and C2 — the paper's second stated contribution — is not testable from the reported results.

This matters concretely rather than pedantically, because there is a specific and plausible competing explanation. E_eff places "this object closes an open goal" and "this object undoes something true" one hop from the ranked quantity; the goal-relevance bit marks objects appearing in open goal atoms. On Visitall — a single-schema movement domain where the goal is a set of visited atoms — those two features very nearly *are* a competent policy, independent of any producer–consumer reasoning. The 4% → 68% jump is therefore consistent with either the chain layer or the goal-directed features doing the work, and the paper's own achievement-oracle diagnostic (designed to rule out terminal-trigger behavior) cannot separate them, because GADAR contains E_eff while the ablation that would test it does not exist.

**Do the results establish the contributions?** C1: yes. C2: not established, for the reason above. C3: established on one domain out of three measured. Miconic reaches 0% on the hard split and Grid 2.08%; the paper reports both honestly as partial transfers, which is to its credit, but the abstract, introduction, and conclusion all lead with the Visitall number as though it characterized the method. A single-domain success with two adjacent near-failures is a promising signal, not a demonstration of transfer.

---

## STRENGTHS

**S1. The problem is real, correctly identified, and correctly positioned.** The observation that shaping the network to its domain is simultaneously what aids within-domain generalization and what makes the policy unexecutable elsewhere is a genuine and useful framing. The related-work section places the contribution accurately: it credits GOOSE and STRIPS-HGN for domain-agnostic *inputs*, identifies the open gap as the *output* side, and cites the specific prior statement that lifted policy learning remains open.

**S2. C1 is achieved and the mechanism is clearly explained.** The design discipline — every feature a flag or one-hot, no dimension sized by |A|, |P|, |T|, or |O|, selection always a dot product against a graph node — is consistently maintained and adequate to the claim. Renaming invariance by construction, verified as a unit test on input tensors, is the right way to establish that property, and it is a real robustness advantage.

**S3. The ladder is the right *shape* of experiment.** Holding encoder, decoder, data, and executor fixed and varying only the representation makes the ablation and the system comparison the same object. The UNION control is exactly the right control: it establishes that the gap is representational rather than architectural or data-driven, and its collapse off the training domains (never above 1% on any held-out split, top-ranked proposal applicable in at most 0.2% of states) is a clean and convincing negative result.

**S4. Several methodological choices are unusually well argued.** The justification for bucketing goal distance rather than using a numeric value — that a number must be read on a scale and no scale is shared across domains — is a genuine insight, and the same argument correctly motivates local action ranking over value regression. The convergent evidence cited from Müller et al. (2026) strengthens this.

**S5. Careful controls against the obvious deflationary readings.** Reporting the untrained control both with and without the revisit monitor (50.4% / 8.0% for GADAR versus 20.8% / 0.0% untrained, under pure greedy execution) is the right response to the "the executor is solving it" concern, and the pure-greedy numbers should be given more prominence than they currently receive.

**S6. Honest reporting of costs and limits.** The in-domain table showing 60.5% against per-domain GABAR's 92.5%, the explicit naming of the "expressiveness tax" on Miconic, the admission that the mechanism behind UNION's bimodality has not been isolated, the framing of Miconic and Grid as partial transfers, and the identification of structural isomorphism with differing semantics as a precise ceiling — these are all creditable and make the paper easier to trust than its headline numbers alone would.

---

## WEAKNESSES

**W1. The ablation does not isolate either claimed novel relation, contrary to explicit statements in the paper.** As detailed above, the GADAR-BIND → GADAR step bundles E_chain, E_eff, goal distance, and goal relevance; the UNION → GADAR-BIND step bundles symbol-identity removal, the lifted layer, and E_bind. Since C2 names binding and chain edges as the contribution, and Table 1 is offered as the ablation study, the paper's second contribution currently has no supporting evidence. The fix is two to four additional runs (GADAR-BIND + E_chain only; GADAR − E_chain; and ideally GADAR-BIND − E_bind), which is well within reasonable resource bounds given that a full eight-domain model trains in seven to eight hours. This is the single most consequential revision.

**W2. Checkpoint selection uses the held-out domain, which compromises the zero-shot claim.** The setup states that zero-shot results "evaluate every saved checkpoint on the held-out domain and report the best." Applying the same rule to every rung preserves the *relative* comparison, but it invalidates the absolute numbers as zero-shot: a genuinely zero-shot system has no access to the target domain for any decision, including model selection. The headline 68% is therefore an oracle-selected best-of-N rather than a zero-shot result. Reporting the number obtained by the in-domain selection rule (combined training and validation loss on the seven training domains) is necessary, and the gap between the two is itself informative.

**W3. The flagship result rests on a single domain, and that domain is plausibly the most favorable one available.** Of six held-out cells measured, one is a success (Visitall H, 68%), one is moderate (Visitall E, 100%, on a split where an untrained network already reaches 26.4%), one is partial (Miconic E, 35.96%), and three are at or near zero. Visitall is a single-schema movement domain whose goal structure is directly encoded by the goal-relevance bit and E_eff. Moreover, the training set retains Grid and Logistics, both of which contain movement schemas structurally close to Visitall's, so the held-out domain is arguably a near-duplicate of training structure rather than a novel one; the same concern applies in reverse to the Gripper/Miconic pair, which the paper itself identifies as the same producer–consumer structure. A short analysis of structural similarity between each held-out domain and its training set would either substantially strengthen the transfer claim or correctly qualify it.

**W4. Table 1 is incomplete, and the gaps are exactly where the claim is weakest.** Five of eight domains are never used as held-out targets, and the table carries "n/a" cells. The paper is right that n/a is not zero, but with three targets reported and two of them near-failures, the unmeasured five carry a great deal of weight. Running the remaining leave-one-out folds — the protocol is already implemented and each fold is a single training run — would convert a suggestive result into a characterization of *when* transfer occurs. As it stands, a reader cannot tell whether Visitall is representative or exceptional.

**W5. The closest prior system is never compared against empirically.** GOOSE's domain-independent setting trains one heuristic across domains and evaluates on held-out ones; the paper leans on it throughout the framing but reports no number from it. The stated distinction — that heuristics delegate action construction to search — is somewhat rhetorical, since a learned heuristic applied greedily over generated successors is itself a search-free policy in the relevant sense, and would be a natural and informative baseline on precisely these held-out splits. Without it, the claim of a *first* domain-agnostic policy is a claim about framing rather than about capability. Either the comparison should be run, or the paper should argue explicitly why greedy hill-climbing on a domain-independent learned heuristic is not an admissible baseline.

**W6. No runtime or grounding cost is reported, which weakens the "without search" claim.** By the cost analysis, E_bind, E_eff, and the goal marks are per-state and linear in the *applicable groundings*, meaning the graph construction requires enumerating all applicable ground actions at every state. That enumeration is the expensive part of grounding and is potentially exponential in schema arity. Since the paper's stated advantage over classical planning is that generality is bought without search, per-decision wall-clock and per-state grounding cost — set against a planner on the same instances — are needed to substantiate it. The current text reports only training time.

**W7. No error analysis.** The evaluation guidelines call for it, and the paper's own results make it necessary. For the failing cells, the decomposition of failures into (a) top-ranked action inapplicable, (b) dead end reached, (c) step bound exhausted, and (d) revisit-monitor thrashing would be highly informative, and the top-1 applicability rate — reported only for UNION, at ≤0.2% — should be reported for every rung on every split. On Miconic the paper offers a plausible hypothesis (per-step errors compounding over a 182-step horizon) but no measurement; a plot of success rate against expert plan length would test it directly.

**W8. The achievement-oracle diagnostic is underspecified and may not bound what it claims.** The oracle "prefers any applicable ground action whose add effects directly satisfy an unsatisfied goal atom, and otherwise defers to the policy" — but which policy it defers to is not stated. If it defers to an untrained network, the 50% figure is not a ceiling on hand-coded terminal triggers combined with competent behavior, and the claim that GADAR "exceeds this ceiling by a factor of two" does not follow. Additionally, since GADAR itself contains E_eff, which places goal-achieving effects one hop from the ranked quantity, the cleaner version of this diagnostic is GADAR with E_eff removed.

**W9. Statistical reporting is thin.** Three seeds are averaged but no standard deviation, range, or per-seed number appears anywhere. Coverage percentages are reported to two decimal places in some cells (35.96, 11.98, 2.08) and one in others, and the instance count per split is never given, so a reader cannot recover how many problems a percentage represents. With three seeds and splits that appear to contain a few dozen instances, differences of a few points are not interpretable without dispersion.

**W10. The claim that the chain layer "adds no learned parameters" should be checked.** In most relational GNN implementations, introducing new edge types widens the relation-type indicator in R and therefore adds parameters to the edge-update function. If the implementation genuinely holds parameter count fixed, the mechanism should be stated; if not, the claim — which appears in the abstract, the results discussion, and the conclusion — should be softened to "no new supervision and no domain-specific parameters."

**W11. Scope of the PDDL fragment is not stated up front.** Negative preconditions, conditional effects, equality, constants declared in D, and axioms are never discussed; "richer PDDL fragments" appears only in the final sentence as future work. Since the contribution is a claim about handling arbitrary domain descriptions, the supported fragment belongs in the problem setup, not in the closing paragraph.

**W12. Presentation.** The prose is dense and heavily figurative in places ("classical planners buy generality with search and learned policies buy reactivity with specialization"), which makes several technically important sentences slower to parse than they need to be. Table 1's layout, with the split column embedded in the domain column, is hard to read. The untrained control is central to the argument but appears only in running text and should be a column in Table 1. Figure 1 carries a great deal of information and would benefit from a reduced version showing only the binding and chain edges under discussion.

---

## RATING

**4 / 10** — below the acceptance threshold, but the core idea is worth developing and several of the required fixes are within reach.

---

## JUSTIFICATION OF RECOMMENDATION

The paper identifies a real and well-motivated gap, proposes a design that plausibly closes it, and executes the capability claim (C1) cleanly: the demonstration that a single policy with no domain-sized component can emit well-formed grounded actions in an unseen domain is genuine, and renaming invariance by construction is a real property, correctly argued and correctly verified. The UNION control is well designed and its collapse is convincing. The paper is also unusually candid about its costs and limits, which makes it pleasant to review.

The recommendation is nonetheless below threshold for three reasons, in order of weight.

First, the paper's second stated contribution is not supported by any experiment. Binding and chain edges are named as the technical novelty, and the ladder is offered as the study that isolates them, but both adjacent steps are bundles of three or four changes. The 4% → 68% result is the paper's centerpiece, and there is a specific, plausible alternative explanation for it — the grounded effect edges and goal-relevance features, which encode goal-directed one-step information on a domain where that information nearly suffices. Until the confound is broken, the paper demonstrates that *something* in the last rung produces transfer on Visitall, not that the chain layer does.

Second, the zero-shot protocol selects checkpoints on the held-out domain. This is the sort of leakage that a reader will discount heavily once noticed, and it applies to every headline number.

Third, the empirical base is too narrow for the generality claimed. Three of eight domains measured, one success, two near-failures, no dispersion, no instance counts, no error analysis, and no comparison to the domain-independent learned-heuristic line that the framing depends on. The abstract and conclusion generalize from the Visitall cell in a way the rest of the table does not license.

The distinguishing consideration is that these are, in the main, deficiencies of experimental design and reporting rather than of the idea. The missing ablations are a small number of additional runs on infrastructure that already exists; the remaining leave-one-out folds likewise; the checkpoint-selection issue requires only re-reporting under an already-implemented rule; and a formal expressiveness statement for the chain layer is a modest theoretical addition that would materially strengthen the paper. A revision addressing W1–W4 with the remaining folds and a de-confounded ablation would be a substantially different and considerably more persuasive submission. If the rebuttal supplies the isolating ablation and the training-domain-selected checkpoint numbers, and they hold up, an upward revision to 5–6 would be warranted.

---

## SPECIFIC POINTS OF FEEDBACK FOR REBUTTAL

Ordered by how much the answer would change the assessment.

1. **De-confound the chain layer (essential).** Please report two additional configurations on the held-out splits: (a) GADAR-BIND + E_chain only, without E_eff, goal distance, and goal relevance; and (b) full GADAR with E_chain removed but E_eff, goal distance, and goal relevance retained. If the 68% on Visitall H survives (a) and drops under (b), the paper's central claim is established. If it survives (b), the claim needs restating in terms of goal-directed grounded features rather than the chain layer.

2. **Report zero-shot numbers under training-domain checkpoint selection (essential).** What is coverage on each held-out split when the checkpoint is chosen by combined training and validation loss over the seven training domains, as in the in-domain protocol? How large is the gap to the best-checkpoint numbers in Table 1?

3. **Complete the leave-one-out matrix.** Results for the five unreported held-out domains (Blocks, Spanner, Gripper, Logistics, Rovers), even at reduced instance counts, would establish whether Visitall is representative. If a subset is infeasible before the rebuttal deadline, please indicate which folds have been run and what they show.

4. **Report dispersion and instance counts.** Per-seed values or standard deviations across the three seeds for every cell in Table 1, and the number of instances in each split. Are the differences between GADAR-BIND and GADAR on Grid (3.1% versus 11.98% on E) larger than seed variance?

5. **Isolate the binding layer.** The UNION → GADAR-BIND step changes three things at once. What is coverage for a variant with structural features and the lifted layer but *without* E_bind? This is the only evidence that would support binding edges as a distinct contribution.

6. **Clarify the achievement oracle.** When the oracle does not fire, which policy does it defer to — untrained, UNION, GADAR-BIND, or GADAR? The interpretation of the 50% ceiling depends entirely on this. Additionally, what does GADAR without E_eff achieve on easy Visitall?

7. **Address the domain-independent heuristic baseline.** Would greedy hill-climbing over generated successors using a heuristic trained in GOOSE's domain-independent setting be admissible as a search-free baseline on these held-out splits? If it is judged inadmissible, an explicit argument would strengthen the "first domain-agnostic policy" claim considerably. If a number can be obtained, it would be the most valuable single addition after point 1.

8. **Report decision-time cost.** Mean per-decision wall-clock time and mean number of applicable groundings enumerated per state, on both splits of at least one held-out domain, alongside the corresponding figures for the reference planner. Does the per-state grounding requirement bound the instance sizes at which the approach is usable?

9. **Provide an error decomposition.** For the failing cells, the breakdown of terminations into top-ranked-action-inapplicable, dead end, and step-bound exhaustion, plus top-1 applicability rate per rung per split. For Miconic specifically, a plot of success rate against expert plan length would test the compounding-error hypothesis directly.

10. **Confirm the parameter-count claim.** Does adding E_chain and E_eff leave the total learned parameter count unchanged, given that R concatenates relation-type indicators? If the edge-update function widens, the phrasing in the abstract and conclusion should be adjusted.

11. **State the supported PDDL fragment in the problem setup.** Specifically: negative preconditions, conditional effects, equality, constants in D, derived predicates. Which of the eight benchmark domains, if any, required preprocessing to fit the supported fragment?

12. **Consider a formal statement for the chain layer.** A construction exhibiting two tasks indistinguishable to L rounds of message passing on the graph without E_chain and distinguishable with it would place the paper's central intuition on the same footing as Theorem 4.3 of Chen, Thiébaux, and Trevizan (2024), which the paper already invokes for the analogous claim about LLG.

13. **Characterize structural distance between held-out and training domains.** Given that Gripper and Miconic are described in the paper as the same producer–consumer structure, and that Grid and Logistics both contain movement schemas resembling Visitall's, a measure of structural overlap per fold would clarify whether the reported transfer is to a novel structure or to a near-duplicate of one seen in training.

14. **Move the untrained control into Table 1** as a full column on both splits, and give the pure-greedy (monitor-removed) numbers equal prominence — they are the more convincing evidence and are currently buried in prose.
