"""Batched GRPO rollouts: the per-row stop rule and lockstep token bookkeeping.

No torch, no model. Tokens are characters (decode = chr join), so a turn's
token ids are exactly its text and every assertion reads directly.
"""

from training.common import row_finished, run_episodes_batched

EOS = 0


def ids(text: str) -> list[int]:
    return [ord(c) for c in text]


def decode(tokens) -> str:
    return "".join(chr(t) for t in tokens if t != EOS)


def done(text: str) -> bool:
    return row_finished(ids(text), decode, {EOS})


# ------------------------------------------------------------------ stop rule
def test_inspect_needs_a_character_past_the_number():
    # Stopping on "INSPECT 1" would cut off a model about to write 12.
    assert not done("HYPOTHESIS: x\nACTION: INSPECT 1")
    assert done("HYPOTHESIS: x\nACTION: INSPECT 12\n")


def test_confidence_decimal_point_does_not_end_the_number():
    assert not done("ACTION: VERDICT AI confidence=0.")
    assert not done("ACTION: VERDICT AI confidence=0.8")
    assert done("ACTION: VERDICT AI confidence=0.85\n")


def test_eos_ends_a_turn_without_an_action():
    assert row_finished(ids("rambling") + [EOS], decode, {EOS})


def test_short_or_empty_generation_never_stops():
    assert not row_finished([], decode, {EOS})
    assert not done("ACTION")


def test_action_at_the_end_of_a_long_turn_is_still_found():
    # The stopper decodes only a tail window. A turn's ACTION line is always at
    # the newest tokens when it completes, so a 2,000-character preamble must
    # not hide it.
    assert done("REASONING: " + "x" * 2000 + "\nACTION: INSPECT 7\n")


def test_prose_mentioning_inspect_without_action_label_does_not_stop():
    assert not done("REASONING: I would INSPECT 6 next because the edge looks smeared ")


# ------------------------------------------------------- lockstep bookkeeping
class FakeEnv:
    """Episode for index i takes i+1 turns: i INSPECTs, then a VERDICT."""

    def reset(self, options):
        self.index, self.turn = options["index"], 0
        return self._obs(), {}

    def _obs(self):
        return {"messages": [], "images": [], "index": self.index, "turn": self.turn}

    def step(self, text):
        self.turn += 1
        finished = "VERDICT" in text
        info = {"action_type": "verdict" if finished else "inspect",
                "correct": finished, "steps": self.turn, "index": self.index}
        return self._obs(), (1.0 if finished else -0.05), finished, False, info


class FakePolicy:
    def __init__(self):
        self.batch_sizes = []

    def act_batch(self, obs_list, *, collect_tokens=False, collect_logprobs=False, **_):
        self.batch_sizes.append(len(obs_list))
        out = []
        for o in obs_list:
            text = "V" if o["turn"] >= o["index"] else "I"
            text = "ACTION: VERDICT AI confidence=0.9\n" if text == "V" else f"ACTION: INSPECT {o['turn'] + 1}\n"
            out.append({
                "text": text,
                # The prompt grows every turn; only the first turn's belongs in the result.
                "prompt_ids": [100 + o["index"]] * (o["turn"] + 1) if collect_tokens else [],
                "token_ids": ids(text) if collect_tokens else [],
                "logprobs": [-0.5] * len(text) if collect_logprobs else [],
            })
        return out


def run(jobs, n_envs, **kw):
    policy = FakePolicy()
    envs = [FakeEnv() for _ in range(n_envs)]
    res = run_episodes_batched(policy, envs, [(i, "clean") for i in jobs], max_turns=10,
                               sample=True, collect_tokens=True, **kw)
    return res, policy


def test_results_come_back_in_job_order_despite_uneven_lengths():
    # GRPO groups consecutive results, so order must follow the jobs even though
    # the 1-turn episode finishes long before the 4-turn one.
    res, _ = run([3, 0, 2, 1], n_envs=4)
    assert [r["index"] for r in res] == [3, 0, 2, 1]
    assert [r["steps"] for r in res] == [4, 1, 3, 2]


def test_prompt_is_first_turn_only_and_completion_spans_all_turns():
    res, _ = run([2], n_envs=1)
    ep = res[0]
    assert ep["prompt_ids"] == [102]
    expected = "ACTION: INSPECT 1\nACTION: INSPECT 2\nACTION: VERDICT AI confidence=0.9\n"
    assert decode(ep["completion_ids"]) == expected
    assert ep["logprobs"] == []
    assert ep["episode_reward"] == 1.0 - 0.05 * 2


def test_logprobs_align_with_completion_tokens_when_requested():
    res, _ = run([1, 3], n_envs=2, collect_logprobs=True)
    for ep in res:
        assert len(ep["logprobs"]) == len(ep["completion_ids"])


def test_slots_refill_when_jobs_outnumber_envs():
    res, policy = run([0, 0, 0, 0, 0, 0], n_envs=4)
    assert len(res) == 6
    assert max(policy.batch_sizes) == 4      # never more live episodes than envs
    assert sum(policy.batch_sizes) == 6      # each 1-turn episode generated once


def test_batch_shrinks_as_episodes_finish():
    _, policy = run([0, 1, 2, 3], n_envs=4)
    assert policy.batch_sizes == [4, 3, 2, 1]
