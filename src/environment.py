"""Custom Gymnasium environment for active-perception forgery detection.

An episode presents one image from the manifest. The agent (a VLM) investigates
it over several turns by zooming into 3x3 grid cells and requesting forensic
metadata, then commits to an AI/REAL verdict. Reward favors a correct verdict
reached with few wasted actions.

Observation note
----------------
This is a multimodal, conversational environment, which the standard Gymnasium
``spaces`` primitives don't capture well. Each observation is a dict::

    {
        "messages": [ {role, content:[{type:"image"} | {type:"text", text}]}... ],
        "images":   [PIL.Image, ...],   # in the order their placeholders appear
        "step":     int,
        "max_steps": int,
    }

``messages`` + ``images`` are exactly what a HuggingFace VLM processor consumes.
The declared ``observation_space`` only describes the scalar bookkeeping fields;
treat the dict above as the real contract.

Smoke test: ``python src/environment.py`` runs a scripted policy and prints the
transitions, which is the verification command referenced in CLAUDE.md.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

# Allow `python src/environment.py` (script context) as well as
# `python -m src.environment` / `from src import ...` (package context).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from src import prompt_templates as pt
from src import utils


@dataclass
class RewardConfig:
    """Reward shaping. Defaults are tuned so that a correct verdict dominates,
    while a per-step cost nudges the policy toward efficient investigation
    rather than burning the whole turn budget every episode."""

    correct: float = 1.0
    incorrect: float = -1.0
    step_cost: float = -0.05          # charged on every non-terminal action
    invalid_penalty: float = -0.10    # malformed / unparseable action
    repeated_zoom_penalty: float = -0.10
    no_answer_penalty: float = -1.0   # ran out of turns without deciding
    format_bonus: float = 0.0         # optional reward for a well-formed action
    # Penalty applied to a CORRECT verdict that was reached after using METADATA.
    # Pushes the policy to discriminate from pixels alone: image-only correct
    # answers out-score metadata-assisted ones, while both still beat a wrong
    # answer. Set to 0.0 to disable.
    metadata_correct_penalty: float = -0.5

    # --- Option A: reasoning-presence floor ------------------------------- #
    # Small per-turn bonus paid whenever the turn carries a substantive,
    # non-boilerplate THOUGHT (see utils.is_substantive_thought). Keeps the
    # reasoning channel from collapsing under outcome-only credit assignment.
    # It rewards *having* a reason, not its correctness -- that is Option C.
    reasoning_bonus: float = 0.05
    reasoning_min_words: int = 4      # min words for a thought to count

    # --- Option C: judged reasoning quality (STUBBED) --------------------- #
    # When enabled, a separate judge scores the final-answer reasoning for
    # grounding/coherence and its score (in [0, 1]) is scaled by
    # ``reasoning_judge_weight`` and added -- but ONLY on a correct verdict, so
    # the judge can never reward an eloquent wrong answer. Requires a judge
    # callable passed to ForgeryDetectionEnv(reasoning_judge=...). Off by
    # default; ships as a no-op stub until a judge is wired in.
    use_reasoning_judge: bool = False
    reasoning_judge_weight: float = 0.2


@dataclass
class EpisodeState:
    record: dict = field(default_factory=dict)
    image: Image.Image | None = None
    messages: list = field(default_factory=list)
    images: list = field(default_factory=list)
    viewed_cells: set = field(default_factory=set)
    metadata_seen: bool = False
    steps: int = 0
    return_so_far: float = 0.0
    trace: list = field(default_factory=list)  # one record per turn


class ForgeryDetectionEnv(gym.Env):
    """Multi-turn active-perception environment over a JSONL image manifest."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        manifest_path: str,
        max_steps: int = 6,
        grid: int = 3,
        upscale_to: int | None = 512,
        reward_config: RewardConfig | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        reasoning_judge: "Callable[[str, Image.Image, str], float] | None" = None,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.max_steps = max_steps
        self.grid = grid
        self.upscale_to = upscale_to
        self.reward_config = reward_config or RewardConfig()
        self.shuffle = shuffle
        # Option C hook: callable(thought, image, verdict) -> score in [0, 1].
        # Left None ships the judge as a no-op stub (see _judge_reasoning).
        self.reasoning_judge = reasoning_judge

        self.records = self._load_manifest(manifest_path)
        if not self.records:
            raise ValueError(f"No records loaded from manifest: {manifest_path}")

        self._rng = np.random.default_rng(seed)
        self._cursor = 0  # used for sequential (non-shuffled) iteration

        # Action is the model's full text completion for the turn.
        self.action_space = spaces.Text(max_length=8192)
        # Only the scalar bookkeeping is expressible as a Gym space; the rich
        # multimodal payload rides along in the observation dict (see module doc).
        self.observation_space = spaces.Dict(
            {
                "step": spaces.Discrete(max_steps + 1),
                "max_steps": spaces.Discrete(max_steps + 1),
            }
        )

        self.state: EpisodeState | None = None

    # ------------------------------------------------------------------ #
    # Manifest loading
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

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        index = (options or {}).get("index")
        if index is None:
            index = self._next_index()
        record = self.records[index]

        image = Image.open(record["file_name"]).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": pt.SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": pt.INITIAL_USER_TEXT},
                ],
            },
        ]

        self.state = EpisodeState(
            record=record,
            image=image,
            messages=messages,
            images=[image],
        )
        return self._observation(), self._info()

    def step(self, action: str):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        s = self.state
        cfg = self.reward_config
        s.steps += 1

        # Record the agent's turn so the conversation stays coherent for the
        # next generation call.
        s.messages.append({"role": "assistant", "content": [{"type": "text", "text": action}]})

        parsed = utils.parse_action(action)
        reward = 0.0
        terminated = False
        truncated = False
        bbox = None          # set for ZOOM, recorded in the trace
        metadata_shown = None  # set for METADATA

        if parsed.type == utils.ANSWER:
            correct = parsed.verdict == utils.label_to_verdict(s.record["label"])
            reward += cfg.correct if correct else cfg.incorrect
            reward += cfg.format_bonus
            # Discourage leaning on the (label-correlated) metadata channel: a
            # correct verdict reached after viewing metadata is penalized so the
            # policy is pushed to decide from the image alone.
            if correct and s.metadata_seen:
                reward += cfg.metadata_correct_penalty
            # Option C: judged reasoning quality, gated on a correct verdict so
            # it can only ever break ties between right answers, never reward an
            # eloquent wrong one. No-op unless enabled + a judge is wired in.
            if correct and cfg.use_reasoning_judge:
                score = self._judge_reasoning(parsed.thought, s.image, parsed.verdict)
                reward += cfg.reasoning_judge_weight * score
            terminated = True
            self._add_user_text(
                f"Verdict recorded: {parsed.verdict}. Investigation complete."
            )

        elif parsed.type == utils.ZOOM:
            reward += cfg.step_cost
            bbox = list(
                utils.grid_cell_bbox(s.image.width, s.image.height, parsed.cell, self.grid)
            )
            if parsed.cell in s.viewed_cells:
                reward += cfg.repeated_zoom_penalty
                self._add_user_text(pt.REPEATED_ZOOM_FEEDBACK.format(cell=parsed.cell))
            else:
                reward += cfg.format_bonus
                s.viewed_cells.add(parsed.cell)
                crop = utils.crop_grid_cell(
                    s.image, parsed.cell, grid=self.grid, upscale_to=self.upscale_to
                )
                s.images.append(crop)
                s.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": pt.ZOOM_RESULT_TEXT.format(cell=parsed.cell)},
                        ],
                    }
                )

        elif parsed.type == utils.METADATA:
            reward += cfg.step_cost + cfg.format_bonus
            s.metadata_seen = True
            metadata_shown = dict(s.record.get("metadata", {}))
            report = pt.format_metadata(metadata_shown)
            self._add_user_text(pt.METADATA_NOTICE.format(report=report))

        else:  # INVALID
            reward += cfg.step_cost + cfg.invalid_penalty
            self._add_user_text(pt.INVALID_ACTION_FEEDBACK)

        # Option A: pay a small bonus for a substantive, non-boilerplate THOUGHT
        # on any well-formed turn, so the reasoning channel survives outcome-only
        # credit assignment. Compared against the previous turn's thought to
        # cheaply discourage copy-paste reasoning.
        if parsed.type != utils.INVALID:
            prev_thought = s.trace[-1]["thought"] if s.trace else ""
            if utils.is_substantive_thought(
                parsed.thought, prev_thought, cfg.reasoning_min_words
            ):
                reward += cfg.reasoning_bonus

        # Enforce the turn budget for non-terminal actions.
        if not terminated:
            if s.steps >= self.max_steps:
                reward += cfg.no_answer_penalty
                truncated = True
            elif s.steps == self.max_steps - 1:
                self._add_user_text(pt.OUT_OF_TURNS_FEEDBACK)

        reward = float(reward)
        s.return_so_far += reward
        s.trace.append(
            {
                "turn": s.steps,
                "thought": parsed.thought,
                "action_type": parsed.type,
                "cell": parsed.cell,
                "bbox": bbox,
                "verdict": parsed.verdict,
                "metadata_shown": metadata_shown,
                "reward": round(reward, 4),
                "cum_reward": round(s.return_so_far, 4),
            }
        )

        return self._observation(), reward, terminated, truncated, self._info(parsed)

    def _judge_reasoning(self, thought: str, image: Image.Image, verdict: str) -> float:
        """Option C (stub): score final-answer reasoning quality in [0, 1].

        Delegates to the ``reasoning_judge`` callable passed at construction --
        typically a VLM-judge that rates whether ``thought`` is grounded in
        ``image``, internally coherent, and entails ``verdict``. Until one is
        wired in this is a no-op returning 0.0, so enabling
        ``use_reasoning_judge`` without a judge changes nothing.

        Note: this runs inside the rollout hot loop (one call per terminated
        episode), so a real judge should be batched/cached at the call site.
        """
        if self.reasoning_judge is None:
            return 0.0
        score = float(self.reasoning_judge(thought, image, verdict))
        # Clamp so a miscalibrated judge can't blow up the advantage scale.
        return max(0.0, min(1.0, score))

    def render(self):
        if self.state is None:
            return
        for msg in self.state.messages:
            text = " ".join(
                c.get("text", "[image]") for c in msg["content"]
            )
            print(f"[{msg['role']}] {text}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _add_user_text(self, text: str):
        self.state.messages.append(
            {"role": "user", "content": [{"type": "text", "text": text}]}
        )

    def _observation(self) -> dict:
        s = self.state
        return {
            "messages": copy.deepcopy(s.messages),
            "images": list(s.images),
            "step": s.steps,
            "max_steps": self.max_steps,
        }

    def _info(self, parsed: utils.Action | None = None) -> dict:
        s = self.state
        info = {
            "id": s.record.get("id"),
            "label": s.record.get("label"),
            "ground_truth": utils.label_to_verdict(s.record["label"]),
            "viewed_cells": sorted(s.viewed_cells),
            "metadata_seen": s.metadata_seen,
            "steps": s.steps,
        }
        if parsed is not None:
            info["action_type"] = parsed.type
            info["predicted_verdict"] = parsed.verdict
            info["correct"] = (
                parsed.verdict == info["ground_truth"] if parsed.is_terminal else None
            )
        return info

    def get_trace(self, global_step: int | None = None, phase: str = "train") -> dict:
        """Serializable record of the just-finished (or in-progress) episode for
        offline replay in the browser visualizer. Call after the episode ends,
        before the next ``reset()``."""
        s = self.state
        truth = utils.label_to_verdict(s.record["label"])
        final = s.trace[-1] if s.trace else {}
        answered = final.get("action_type") == utils.ANSWER
        prediction = final.get("verdict") if answered else None
        return {
            "episode_id": s.record.get("id"),
            "image_path": s.record.get("file_name"),
            "label": s.record.get("label"),
            "ground_truth": truth,
            "prediction": prediction,
            "correct": bool(answered and prediction == truth),
            "answered": answered,
            "total_reward": round(s.return_so_far, 4),
            "num_turns": len(s.trace),
            "viewed_cells": sorted(s.viewed_cells),
            "metadata_seen": s.metadata_seen,
            "grid": self.grid,
            "global_step": global_step,
            "phase": phase,
            "turns": s.trace,
        }


# ----------------------------------------------------------------------- #
# Smoke test: scripted policy, no model required.
# ----------------------------------------------------------------------- #
def _smoke_test():
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = os.path.join(here, "..", "data", "metadata.jsonl")

    env = ForgeryDetectionEnv(manifest_path=manifest, max_steps=6, seed=0)
    obs, info = env.reset()
    print(f"Reset episode id={info['id']} truth={info['ground_truth']} "
          f"images={len(obs['images'])}")

    scripted = [
        "THOUGHT: Let me inspect the center for fine-detail artifacts.\nACTION: ZOOM 5",
        "THOUGHT: Check a corner too.\nACTION: ZOOM 1",
        "THOUGHT: Now verify forensic metadata.\nACTION: METADATA",
        "THOUGHT: Metadata and textures look consistent with a real photo.\nACTION: ANSWER REAL",
    ]

    total = 0.0
    for action in scripted:
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(
            f"step={info['steps']} action={info['action_type']:<8} "
            f"reward={reward:+.2f} images={len(obs['images'])} "
            f"terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            print(f"  -> predicted={info.get('predicted_verdict')} "
                  f"correct={info.get('correct')}")
            break

    print(f"Episode return: {total:+.2f}")


if __name__ == "__main__":
    _smoke_test()
