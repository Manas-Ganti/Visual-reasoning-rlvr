# Reward-failure history

The reward in `env/reward.py` is the fourth iteration. Each earlier version was
killed by a specific, observed failure mode — documenting them is the point, not
an afterthought: it's the evidence that the final shape is *earned*, not guessed.

> All terms are mechanically checkable functions of the structured trajectory
> (`env/trajectory.py`). There is no LLM judge at any version — the failures below
> are about *what a verifiable signal can and can't force*, which is exactly the
> interesting part.

## v1 — binary verdict only
```
R = +1 correct / −1 wrong
```
**What broke:** with a class-imbalanced or even balanced set, the policy collapses
onto whichever verdict is locally easiest and stops investigating. The overview is
uninformative by construction, so a model that never inspects still gets ~chance
reward for free — and chance is a low bar to beat by guessing the majority
behavior. No reasoning, no inspection, no signal about *how* it decided.

## v2 — add a "mentioned reasoning" bonus
```
R = v1  + 0.1 · (produced a THOUGHT / hypothesis field)
```
**What broke:** rewarding the *presence* of reasoning text rewards boilerplate.
The policy learned to emit a fixed, plausible-sounding hypothesis every turn
("the eyes will be asymmetric") regardless of the image, collect the bonus, and
still not use the evidence. Eloquence ≠ investigation. This is why the current
reward **never scores prose** — only the numeric belief, the reconciliation
direction, and the committed verdict.

## v3 — add belief-coherence grounding
```
R = v1  + 0.3 · belief_coherence
```
where `belief_coherence` checks that P(fake) moved in the direction the agent's
own reconciliations imply (CONFIRMED-artifact ⇒ up, REFUTED ⇒ down).
**What broke (partially):** this is faithful — it can't be gamed by boilerplate,
because a belief that lurches against its own stated reconciliation scores zero.
But it was too **sparse**: coherence gives a flat reward to a coherent-but-timid
policy that inspects once and hedges, and offers no pressure toward efficiency or
calibration. Learning stalled — lots of internally-consistent, low-information
episodes.

## v4 — graded process credit + budget + calibration (current)
```
R = +1·verdict_correct
    + 0.30·belief_coherence
    + 0.30·verdict_consistency     (final call follows the accumulated evidence)
    + 0.10·prediction_tracking     (SOFT: fraction of hypotheses confirmed)
    − 0.05·per_inspect             (budget pressure → hypothesis-driven looking)
    − 0.50·confident_wrong·conf    (calibration → hedge on indistinguishable images)
    − 1.00·no_answer
```
**Why it holds:** `verdict_correct` dominates so process credit can never rescue a
confidently-wrong episode (see `tests/test_reward.py::test_reward_hacking_guard_incoherent_scores_low`).
`verdict_consistency` adds the missing pressure v3 lacked — the ending must follow
from the middle. The per-inspect cost turns "look at everything" into a losing
strategy, forcing the predict-then-verify discipline. `confident_wrong` scaled by
stated confidence handles the genuinely indistinguishable StyleGAN2 faces:
hedging is correct behavior there, and the reward now says so.

## Reward-hacking surface & mitigations

| Attack | Mitigation |
|---|---|
| **Brute-force the grid** (inspect every cell, no reasoning) | per-inspect `action_cost` + finite budget make exhaustive looking net-negative |
| **Majority-class guess** (never investigate) | uninformative overview + `no_answer`/`confident_wrong` penalties + balanced splits |
| **Boilerplate hypotheses** (fixed reasoning text) | reward reads only numeric belief + reconciliation *direction*, never prose |
| **Always-CONFIRMED** (max out `prediction_tracking`) | that term is SOFT (0.10) and can't move belief-coherence/consistency; the honest check is the eval-only evidence slice (GradCAM + human-verified cells), never the reward |
| **Eloquent wrong answer** | `verdict_correct` dominates; process terms are strictly weaker and `confident_wrong` punishes loud mistakes |

## How we verify the learning is real (Stage 3, `eval/harness.py`)

1. **Held-out degradation** — improvement must appear on a difficulty level never trained on.
2. **Trajectory audit** — read top-reward rollouts in `demo/app.py`: genuine predict-then-verify or verifier-satisfying noise?
3. **Ablation** — drop `belief_coherence`, retrain: does reward inflate while tier-hard accuracy stalls? (catches hacking).
4. **Adversarial probe** — hand-crafted incoherent-but-confident trajectories must score low (unit-tested).
5. **Calibration curve** — does stated 0.8 confidence mean right ~80% of the time?
6. **Evidence slice** — on human-verified fakes, does the RL policy inspect true-artifact cells more than the SFT baseline?
