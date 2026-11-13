"""Full-episode assembly: the GRPO training sequence must match what the policy saw.

Runs 1-3 trained on prompt[0] + the sampled turns glued together, with no env
messages and no inspected cells. These tests pin the replacement: one sequence
that interleaves sampled turns with env messages, an env_mask over it, and an
image list whose length matches the placeholders in the sequence.
"""

import pytest

from training.common import EpisodeAssemblyError, assemble_episode, run_episodes_batched

END, IMG = 1, 2          # turn-end token and one image placeholder token


def turn(prompt, gen, lps=None):
    return {"prompt_ids": prompt, "token_ids": gen, "logprobs": lps or []}


# ------------------------------------------------------------ assemble_episode
def test_interleaves_sampled_turns_with_env_messages():
    p1 = [100, 101, IMG, 102]
    g1, e1 = [10, 11], [END, 50, IMG, 51]
    p2 = p1 + g1 + e1
    g2, e2 = [12, 13], [END, 52, IMG, 53]
    p3 = p2 + g2 + e2
    g3 = [14]
    out = assemble_episode([turn(p1, g1), turn(p2, g2), turn(p3, g3)], END)

    assert out["prompt_ids"] == p1
    assert out["completion_ids"] == g1 + e1 + g2 + e2 + g3
    assert out["env_mask"] == [1, 1] + [0] * 4 + [1, 1] + [0] * 4 + [1]
    assert out["retok_mismatch"] == 0
    # The full sequence is exactly the last prompt plus the last turn: the
    # gradient's context for every turn is what generation saw.
    assert out["prompt_ids"] + out["completion_ids"] == p3 + g3


def test_only_sampled_tokens_are_trainable():
    p1, g1 = [100], [10, 11]
    p2 = p1 + g1 + [END, 50, IMG, 51]
    out = assemble_episode([turn(p1, g1), turn(p2, [12])], END)
    trained = [t for t, m in zip(out["completion_ids"], out["env_mask"]) if m]
    assert trained == [10, 11, 12]
    assert IMG not in trained


def test_turn_that_emitted_turn_end_is_not_duplicated():
    p1, g1 = [100], [10, 11, END]          # model closed its own turn
    p2 = p1 + [10, 11] + [END, 50]
    out = assemble_episode([turn(p1, g1), turn(p2, [12])], END)
    assert out["completion_ids"] == [10, 11, END, 50, 12]
    assert out["completion_ids"].count(END) == 1
    assert out["env_mask"] == [1, 1, 1, 0, 1]


def test_retokenization_mismatch_is_counted_not_hidden():
    p1, g1 = [100], [10, 11]
    p2 = p1 + [10, 99] + [END, 50]          # template re-encoded the text differently
    out = assemble_episode([turn(p1, g1), turn(p2, [12])], END)
    assert out["retok_mismatch"] == 1
    assert out["completion_ids"] == [10, 11, END, 50, 12]   # sampled ids kept


def test_prompt_that_rewrites_history_fails_loudly():
    p1, g1 = [100, 101], [10]
    p2 = [100, 999, 10, END, 50]            # earlier message edited
    with pytest.raises(EpisodeAssemblyError):
        assemble_episode([turn(p1, g1), turn(p2, [12])], END)


def test_missing_turn_end_fails_loudly():
    p1, g1 = [100], [10]
    with pytest.raises(EpisodeAssemblyError):
        assemble_episode([turn(p1, g1), turn(p1 + [10, 50], [12])], END)


def test_logprobs_align_and_env_tokens_get_zero():
    p1, g1 = [100], [10, 11]
    p2 = p1 + g1 + [END, 50, 51]
    out = assemble_episode([turn(p1, g1, [-0.1, -0.2]), turn(p2, [12], [-0.3])], END)
    assert out["logprobs"] == [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3]
    assert len(out["logprobs"]) == len(out["completion_ids"])


def test_logprobs_dropped_when_any_turn_lacks_them():
    p1, g1 = [100], [10]
    out = assemble_episode([turn(p1, g1, [-0.1]), turn(p1 + g1 + [END], [12])], END)
    assert out["logprobs"] == []


def test_single_turn_episode():
    out = assemble_episode([turn([100, IMG], [10, 11])], END)
    assert out["completion_ids"] == [10, 11] and out["env_mask"] == [1, 1]


# ------------------------------------------- end to end through the rollout loop
# A toy chat template over integer tokens: every message is
# [role, *content, END]; the generation prompt is [ASSISTANT]. An image
# content part renders as one IMG token. Prefix-stable, like Qwen's.
SYS, USER, ASSISTANT = 90, 91, 92


def render(messages):
    out = []
    for m in messages:
        out.append({"system": SYS, "user": USER, "assistant": ASSISTANT}[m["role"]])
        for part in m["content"]:
            out.extend([IMG] if part["type"] == "image" else [ord(c) for c in part["text"]])
        out.append(END)
    return out + [ASSISTANT]


class TemplateEnv:
    """INSPECTs ``index`` cells (each reveals one image), then a VERDICT."""

    def reset(self, options):
        self.index, self.turn = options["index"], 0
        self.messages = [{"role": "system", "content": [{"type": "text", "text": "s"}]},
                         {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "go"}]}]
        self.images = ["overview"]
        return self._obs(), {}

    def _obs(self):
        return {"messages": list(self.messages), "images": list(self.images), "turn": self.turn,
                "index": self.index}

    def step(self, text):
        self.turn += 1
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        finished = text.startswith("V")
        if not finished:
            self.images.append(f"cell{self.turn}")
            self.messages.append({"role": "user", "content": [{"type": "image"},
                                                               {"type": "text", "text": "r"}]})
        info = {"action_type": "verdict" if finished else "inspect", "correct": True,
                "steps": self.turn, "index": self.index}
        return self._obs(), 0.0, finished, False, info


class TemplatePolicy:
    def act_batch(self, obs_list, *, collect_tokens=False, collect_logprobs=False, **_):
        out = []
        for o in obs_list:
            text = "V" if o["turn"] >= o["index"] else "I" + str(o["turn"])
            out.append({"text": text, "prompt_ids": render(o["messages"]),
                        "token_ids": [ord(c) for c in text], "logprobs": []})
        return out


def test_rollout_returns_full_episode_with_matching_images():
    envs = [TemplateEnv() for _ in range(3)]
    res = run_episodes_batched(TemplatePolicy(), envs, [(i, "clean") for i in (2, 0, 3)],
                               max_turns=10, collect_tokens=True, full_episode=True,
                               turn_end_id=END)
    for ep, n_inspect in zip(res, (2, 0, 3)):
        seq = ep["prompt_ids"] + ep["completion_ids"]
        # One placeholder per image, overview plus every revealed cell.
        assert seq.count(IMG) == len(ep["images"]) == 1 + n_inspect
        assert ep["images"] == ["overview"] + [f"cell{k}" for k in range(1, n_inspect + 1)]
        assert len(ep["env_mask"]) == len(ep["completion_ids"])
        assert ep["retok_mismatch"] == 0
        assert ep["turns"] == n_inspect + 1
        trained = "".join(chr(t) for t, m in zip(ep["completion_ids"], ep["env_mask"]) if m)
        assert trained == "".join(f"I{k}" for k in range(n_inspect)) + "V"


def test_full_episode_requires_token_collection():
    with pytest.raises(ValueError):
        run_episodes_batched(TemplatePolicy(), [TemplateEnv()], [(0, "clean")], max_turns=2,
                             full_episode=True, turn_end_id=END)
