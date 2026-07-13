"""The investigation environment.

One episode presents one face image. The agent sees only a low-resolution
OVERVIEW at reset (partial observability — fine artifacts are blurred away) and
must spend a limited ``inspect`` budget to reveal 4×4 grid cells at high
resolution before committing a ``verdict``. Because the answer is unreachable
from the overview alone, a correct verdict is evidence that genuine investigation
occurred — which is the whole point of the design.

Only two actions exist:

    INSPECT <n>                       reveal cell n (1..16); costs one budget unit
    VERDICT <AI|REAL> confidence=<c>  terminal

All reasoning (observations, hypotheses, belief updates, reconciliations) lives
in the *text* of each turn, not as extra actions — see ``env/prompts.py`` for the
format and ``env/trajectory.py`` for how it is parsed. The verifiable reward
(``env/reward.py``) is credited once, at episode end, from the accumulated
trajectory; intermediate steps return 0 so the gym per-step sum equals the
episode reward GRPO consumes.

Observation dict (the real contract; ``observation_space`` only declares the
scalar bookkeeping)::

    {
        "messages":  [ {role, content:[{type:"image"} | {type:"text", text}]}... ],
        "images":    [PIL.Image, ...],   # overview first, then each reveal
        "step":      int,
        "inspects_remaining": int,
    }

Smoke test: ``python -m env.environment`` drives a scripted 2-action policy and
prints the transitions + reward breakdown (the CLAUDE.md verification command).
"""

from __future__ import annotations

import copy
import json
import os
import sys

# Allow both ``python -m env.environment`` and ``python env/environment.py``.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from data import degradation as degrade
from env import grid, prompts
from env.reward import RewardConfig, compute_episode_reward
from env.trajectory import INSPECT, INVALID, VERDICT, Trajectory, label_to_verdict, parse_turn


class InvestigationEnv(gym.Env):
    """Multi-turn, image-grounded investigation over a JSONL manifest."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        manifest_path: str,
        max_inspects: int = 4,
        grid_size: int = 4,
        overview_long_edge: int = 140,
        reveal_size: int = 336,
        reward_config: RewardConfig | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        default_degradation: str = "clean",
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.max_inspects = max_inspects
        # A few turns of slack beyond the budget for the terminal verdict turn and
        # a little tolerance for malformed retries, so the episode can't hang.
        self.max_turns = max_inspects + 3
        self.grid_size = grid_size
        self.overview_long_edge = overview_long_edge
        self.reveal_size = reveal_size
        self.reward_config = reward_config or RewardConfig()
        self.shuffle = shuffle
        self.default_degradation = default_degradation

        self.records = self._load_manifest(manifest_path)
        if not self.records:
            raise ValueError(f"No records loaded from manifest: {manifest_path}")

        self._rng = np.random.default_rng(seed)
        self._cursor = 0

        self.action_space = spaces.Text(max_length=8192)
        self.observation_space = spaces.Dict(
            {
                "step": spaces.Discrete(self.max_turns + 1),
                "inspects_remaining": spaces.Discrete(max_inspects + 1),
            }
        )
        self.state: dict | None = None

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_manifest(path: str) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _next_index(self) -> int:
        if self.shuffle:
            return int(self._rng.integers(len(self.records)))
        idx = self._cursor
        self._cursor = (self._cursor + 1) % len(self.records)
        return idx

    def _resolve_path(self, record: dict) -> str:
        """Resolve a (possibly relative) manifest path against the repo root, so
        the manifest is portable regardless of absolute paths baked in at build
        time."""
        p = record["file_name"]
        if os.path.isabs(p) and os.path.exists(p):
            return p
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(root, p)
        return cand if os.path.exists(cand) else p

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        index = options.get("index")
        if index is None:
            index = self._next_index()
        record = self.records[index]
        level = options.get("degradation", self.default_degradation)

        full = Image.open(self._resolve_path(record)).convert("RGB")
        image = degrade.apply(full, level)               # degradation applied ONCE
        overview = grid.make_overview(
            image, long_edge=self.overview_long_edge, restore_to=max(image.size)
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompts.SYSTEM_PROMPT_FULL}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompts.INITIAL_USER_TEXT.format(budget=self.max_inspects)},
                ],
            },
        ]

        self.state = {
            "record": record,
            "index": index,
            "degradation": level,
            "image": image,
            "messages": messages,
            "images": [overview],
            "inspected_cells": set(),
            "inspects_used": 0,
            "turns": 0,
            "trajectory": Trajectory(),
            "trace": [],
            "return": 0.0,
            "done": False,
        }
        return self._observation(), self._info()

    def step(self, action: str):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        s = self.state
        s["turns"] += 1
        s["messages"].append({"role": "assistant", "content": [{"type": "text", "text": action}]})

        entry = parse_turn(action)
        s["trajectory"].add(entry)

        terminated = False
        truncated = False
        executed = False
        bbox = None

        if entry.action_type == VERDICT:
            terminated = True
            executed = True
            self._add_user(
                prompts.VERDICT_ACK_TEXT.format(
                    verdict=entry.verdict,
                    confidence=entry.confidence if entry.confidence is not None else "unspecified",
                )
            )

        elif entry.action_type == INSPECT:
            cell = entry.cell
            if cell is None or not 1 <= cell <= grid.num_cells(self.grid_size):
                self._add_user(prompts.INVALID_ACTION_FEEDBACK)
            elif s["inspects_used"] >= self.max_inspects:
                self._add_user(prompts.BUDGET_EXHAUSTED_TEXT)
            elif cell in s["inspected_cells"]:
                self._add_user(prompts.REPEATED_INSPECT_FEEDBACK.format(cell=cell))
            else:
                executed = True
                s["inspected_cells"].add(cell)
                s["inspects_used"] += 1
                bbox = list(grid.cell_bbox(s["image"].width, s["image"].height, cell, self.grid_size))
                crop = grid.crop_cell(s["image"], cell, self.grid_size, upscale_to=self.reveal_size)
                s["images"].append(crop)
                remaining = self.max_inspects - s["inspects_used"]
                s["messages"].append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompts.INSPECT_RESULT_TEXT.format(cell=cell, remaining=remaining)},
                        ],
                    }
                )
        else:  # INVALID
            self._add_user(prompts.INVALID_ACTION_FEEDBACK)

        # Budget / turn-cap bookkeeping (only relevant if we didn't just answer).
        if not terminated:
            if s["turns"] >= self.max_turns:
                truncated = True
            elif s["inspects_used"] >= self.max_inspects and entry.action_type != INVALID:
                self._add_user(prompts.BUDGET_EXHAUSTED_TEXT)

        reward = 0.0
        if terminated or truncated:
            s["done"] = True
            reward, breakdown = compute_episode_reward(
                s["trajectory"], self._ground_truth(), self.reward_config
            )
            s["reward_breakdown"] = breakdown
            s["return"] = reward

        self._record_trace(entry, executed, bbox)
        return self._observation(), float(reward), terminated, truncated, self._info(entry)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _add_user(self, text: str):
        self.state["messages"].append({"role": "user", "content": [{"type": "text", "text": text}]})

    def _ground_truth(self) -> str:
        return label_to_verdict(self.state["record"]["label"])

    def _observation(self) -> dict:
        s = self.state
        return {
            "messages": copy.deepcopy(s["messages"]),
            "images": list(s["images"]),
            "step": s["turns"],
            "inspects_remaining": self.max_inspects - s["inspects_used"],
        }

    def _info(self, entry=None) -> dict:
        s = self.state
        info = {
            "id": s["record"].get("id"),
            "index": s["index"],
            "label": s["record"].get("label"),
            "ground_truth": self._ground_truth(),
            "degradation": s["degradation"],
            "inspected_cells": sorted(s["inspected_cells"]),
            "inspects_used": s["inspects_used"],
            "steps": s["turns"],
        }
        if entry is not None:
            info["action_type"] = entry.action_type
            info["predicted_verdict"] = entry.verdict
            info["confidence"] = entry.confidence
            info["correct"] = (
                entry.verdict == info["ground_truth"] if entry.is_terminal else None
            )
        if s["done"]:
            info["reward_breakdown"] = s.get("reward_breakdown", {})
            info["episode_reward"] = s["return"]
        return info

    def _record_trace(self, entry, executed: bool, bbox):
        self.state["trace"].append(
            {
                "turn": self.state["turns"],
                "action_type": entry.action_type,
                "cell": entry.cell,
                "bbox": bbox,
                "executed": executed,
                "reconciliation": entry.reconciliation,
                "p_fake": entry.p_fake,
                "observation": entry.observation,
                "reasoning": entry.reasoning,
                "hypothesis": entry.hypothesis,
                "verdict": entry.verdict,
                "confidence": entry.confidence,
            }
        )

    def render(self):
        if self.state is None:
            return
        for msg in self.state["messages"]:
            text = " ".join(c.get("text", "[image]") for c in msg["content"])
            print(f"[{msg['role']}] {text}")

    def get_trace(self, global_step: int | None = None, phase: str = "train") -> dict:
        """Serializable episode record for the Gradio trajectory viewer / eval
        logs. Call after the episode ends, before the next ``reset()``."""
        s = self.state
        truth = self._ground_truth()
        final = s["trajectory"].final
        answered = s["trajectory"].answered
        prediction = s["trajectory"].final_verdict
        return {
            "episode_id": s["record"].get("id"),
            "image_path": self._resolve_path(s["record"]),
            "index": s["index"],
            "degradation": s["degradation"],
            "label": s["record"].get("label"),
            "ground_truth": truth,
            "prediction": prediction,
            "confidence": s["trajectory"].final_confidence,
            "correct": bool(answered and prediction == truth),
            "answered": answered,
            "final_p_fake": s["trajectory"].final_belief(),
            "total_reward": round(s["return"], 4),
            "reward_breakdown": {k: round(v, 4) for k, v in s.get("reward_breakdown", {}).items()},
            "num_turns": len(s["trace"]),
            "inspected_cells": sorted(s["inspected_cells"]),
            "grid": self.grid_size,
            "global_step": global_step,
            "phase": phase,
            "turns": s["trace"],
        }


# ----------------------------------------------------------------------- #
# Smoke test: scripted 2-action policy, no model required.
# ----------------------------------------------------------------------- #
def _synthetic_manifest() -> str:
    """Write a tiny throwaway manifest + generated images so the smoke test runs
    anywhere, with or without the real dataset downloaded."""
    import tempfile

    tmp = tempfile.mkdtemp(prefix="vrr-smoke-")
    rows = []
    for i, label in enumerate((1, 0)):  # one AI, one REAL
        img = Image.new("RGB", (300, 300))
        px = img.load()
        for y in range(300):
            for x in range(300):
                # cheap deterministic texture so overview vs crop actually differ
                px[x, y] = ((x * 3 + label * 40) % 256, (y * 3) % 256, ((x + y) * 2) % 256)
        path = os.path.join(tmp, f"img_{i}.png")
        img.save(path)
        rows.append({"id": f"smoke_{i}", "file_name": path, "label": label, "split": "test"})
    manifest = os.path.join(tmp, "manifest.jsonl")
    with open(manifest, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return manifest


def _smoke_test():
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = os.path.join(here, "..", "data", "manifest.jsonl")
    if not os.path.exists(manifest):
        print("No data/manifest.jsonl yet; using a synthetic manifest for the smoke test.")
        manifest = _synthetic_manifest()

    env = InvestigationEnv(manifest_path=manifest, max_inspects=4, seed=0, shuffle=False)
    obs, info = env.reset(options={"index": 0, "degradation": "clean"})
    print(f"Reset: id={info['id']} truth={info['ground_truth']} "
          f"images={len(obs['images'])} budget={obs['inspects_remaining']}")

    scripted = [
        "OBSERVATION: Blurry centered face.\nREASONING: Eyes leak GAN artifacts first.\n"
        "HYPOTHESIS: If AI, the left iris in cell 6 will be malformed.\nACTION: INSPECT 6",
        "RECONCILIATION: CONFIRMED - the iris edge is warped as predicted.\n"
        "BELIEF_UPDATE: P(fake)=0.75 because the artifact is clear.\n"
        "OBSERVATION: Warped iris.\nREASONING: Corroborate with the ear.\n"
        "HYPOTHESIS: If AI, the earring in cell 8 is asymmetric.\nACTION: INSPECT 8",
        "RECONCILIATION: CONFIRMED - earring halves mismatch.\n"
        "BELIEF_UPDATE: P(fake)=0.88 because two independent artifacts agree.\n"
        "OBSERVATION: Mismatched earring.\nREASONING: Enough evidence to commit.\n"
        "HYPOTHESIS: none.\nACTION: VERDICT AI confidence=0.85",
    ]

    for action in scripted:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  turn={info['steps']} act={info['action_type']:<7} "
              f"exec_reward={reward:+.3f} images={len(obs['images'])} "
              f"remaining={obs['inspects_remaining']} term={terminated} trunc={truncated}")
        if terminated or truncated:
            print(f"  -> predicted={info.get('predicted_verdict')} correct={info.get('correct')}")
            print("  reward breakdown:")
            for k, v in info.get("reward_breakdown", {}).items():
                print(f"      {k:<20} {v:+.3f}")
            print(f"  episode_reward={info.get('episode_reward'):+.3f}")
            break


if __name__ == "__main__":
    _smoke_test()
