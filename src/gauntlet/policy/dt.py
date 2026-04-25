"""Decision Transformer (offline-RL) :class:`Policy` adapter.

Backlog item B-16. Currently the only "trained" baselines users can
compare against are SOTA VLAs (OpenVLA, SmolVLA, π0, GR00T-N1, RDT).
A small Decision Transformer (Chen et al., NeurIPS 2021) trained from
``Runner(trajectory_dir=...)`` parquet dumps is the canonical
"what does a tiny model trained on your own data do?" baseline — it
closes the loop on the trajectory-dump feature (B-23) without
introducing the SOTA-VLA embodiment / licensing baggage.

DT is on Hugging Face as :class:`transformers.DecisionTransformerModel`;
pre-trained Gym checkpoints exist (e.g.
``edbeeching/decision-transformer-gym-hopper-medium``). For gauntlet's
TabletopEnv there is no public checkpoint — that's expected. The
adapter accepts any DT checkpoint via ``model_id`` and the user
supplies their own (trained externally; see the example for the
trajectory-dump → train workflow).

**Anti-feature** (B-16 spec): this PR ships the *adapter only*. There
is NO training surface in :mod:`gauntlet.policy.dt`. Training a DT is
the user's responsibility — gauntlet has explicitly stayed out of the
training story.

Everything torch / transformers / huggingface_hub is imported
**lazily** inside :meth:`DecisionTransformerPolicy.__init__`. Importing
this module with the ``[dt]`` extra uninstalled is fine — instantiating
the class is what raises ``ImportError(_DT_INSTALL_HINT)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gauntlet.policy.base import Action, Observation

if TYPE_CHECKING:  # pragma: no cover — strings at runtime per `from __future__`.
    import torch  # noqa: F401

__all__ = ["DecisionTransformerPolicy"]


_DT_INSTALL_HINT = (
    "DecisionTransformerPolicy requires the 'dt' extra. Install with:\n"
    "    uv sync --extra dt\n"
    "or, for a plain pip env:\n"
    "    pip install 'gauntlet[dt]'"
)

# Default observation key for the proprioceptive state vector. DT
# operates on a flat state representation — :meth:`_state_from_obs`
# concatenates these in order. A real deployment with a custom-trained
# checkpoint overrides ``_DEFAULT_STATE_KEYS`` by subclassing.
_DEFAULT_STATE_KEYS: tuple[str, ...] = ("state",)


class DecisionTransformerPolicy:
    """Policy adapter for Decision Transformer offline-RL checkpoints.

    Wraps :class:`transformers.DecisionTransformerModel`. DT is an
    autoregressive transformer trained on
    ``(return-to-go, state, action)`` triples; at inference it consumes
    a context of the most recent ``context_length`` triples and
    predicts the next action conditioned on the user-supplied target
    return.

    Parameters
    ----------
    model_id:
        HF Hub repo ID or local checkpoint path. Required — there is
        no canonical default DT checkpoint for TabletopEnv (see
        module docstring).
    device:
        Torch device string. Defaults to ``"cpu"``.
    target_return:
        The return-to-go conditioning value DT uses to steer its
        predicted actions. Higher values bias towards more optimal
        trajectories (per the DT paper §4); the appropriate magnitude
        depends on the reward scale of the trajectories the
        checkpoint was trained on. Default 100.0 — a starting point;
        users should sweep this against their own checkpoint.
    context_length:
        Maximum number of past ``(state, action, return-to-go)``
        tuples kept in the rolling buffer. Defaults to 20 (the value
        used by the original DT Gym checkpoints). Must be ``>= 1``.

    Raises
    ------
    ImportError: if the ``[dt]`` extra is not installed.
    KeyError: on ``act`` if the configured state key(s) are missing.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        target_return: float = 100.0,
        context_length: int = 20,
    ) -> None:
        try:
            import torch
            from transformers import DecisionTransformerModel
        except ImportError as exc:
            raise ImportError(_DT_INSTALL_HINT) from exc

        if context_length < 1:
            raise ValueError(f"context_length must be >= 1; got {context_length}")

        self._torch: Any = torch
        self.model_id = model_id
        self.device = device
        self.target_return = float(target_return)
        self.context_length = int(context_length)

        # FFI seam: transformers' ``from_pretrained`` is untyped — bind
        # via ``Any`` (same pattern as RdtPolicy / HuggingFacePolicy —
        # spec §6 carve-out for FFI boundaries). Decision Transformer
        # ships in-tree (no ``trust_remote_code`` needed).
        loader: Any = DecisionTransformerModel.from_pretrained
        model: Any = loader(model_id)
        self._model: Any = model.to(device)
        self._model.eval()

        config: Any = self._model.config
        self.state_dim = int(config.state_dim)
        self.act_dim = int(config.act_dim)

        # Rolling context buffers — populated by :meth:`act`, flushed by
        # :meth:`reset`. Each entry is a 1-D float32 numpy array.
        self._states: list[np.ndarray[Any, Any]] = []
        self._actions: list[np.ndarray[Any, Any]] = []
        self._returns_to_go: list[float] = []
        self._timesteps: list[int] = []
        # Absolute step counter — independent of buffer length so a
        # rolling context preserves the true positional index DT needs
        # for its time embeddings (the buffer truncation drops *old*
        # entries; the *current* step's timestep keeps incrementing).
        self._step: int = 0

    # ---- public API ------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """Run one DT inference step against the rolling context buffer.

        Maintains a buffer of the most recent
        ``(state, action, return-to-go, timestep)`` tuples (capped at
        :attr:`context_length`) and asks the underlying DT to predict
        the next action conditioned on that context plus the
        user-supplied :attr:`target_return`.
        """
        torch = self._torch
        state = self._state_from_obs(obs)

        # Append a placeholder action for the *current* step (DT's
        # autoregressive setup expects an action slot at the
        # to-be-predicted index; it's masked out at inference).
        self._states.append(state)
        self._actions.append(np.zeros(self.act_dim, dtype=np.float32))
        rtg = self.target_return if not self._returns_to_go else self._returns_to_go[-1]
        self._returns_to_go.append(rtg)
        self._timesteps.append(self._step)
        self._step += 1

        self._truncate_context()

        states_t = torch.from_numpy(np.stack(self._states)).unsqueeze(0).to(self.device)
        actions_t = torch.from_numpy(np.stack(self._actions)).unsqueeze(0).to(self.device)
        rtg_t = (
            torch.tensor(self._returns_to_go, dtype=torch.float32).reshape(1, -1, 1).to(self.device)
        )
        ts_t = torch.tensor(self._timesteps, dtype=torch.long).reshape(1, -1).to(self.device)

        with torch.no_grad():
            outputs: Any = self._model(
                states=states_t,
                actions=actions_t,
                returns_to_go=rtg_t,
                timesteps=ts_t,
                return_dict=True,
            )

        action_pred: Any = outputs.action_preds[0, -1]
        if hasattr(action_pred, "detach"):
            action_pred = action_pred.detach()
        if hasattr(action_pred, "to"):
            action_pred = action_pred.to(device="cpu", dtype=torch.float32)
        if hasattr(action_pred, "numpy"):
            arr = np.asarray(action_pred.numpy(), dtype=np.float64).reshape(-1)
        else:
            arr = np.asarray(action_pred, dtype=np.float64).reshape(-1)

        # Replace the placeholder action with the predicted one so the
        # next ``act`` call sees the correct autoregressive history.
        self._actions[-1] = arr.astype(np.float32, copy=False)
        return arr

    def reset(self, rng: np.random.Generator) -> None:
        """Flush the rolling context buffer and reset return-to-go."""
        del rng
        self._states.clear()
        self._actions.clear()
        self._returns_to_go.clear()
        self._timesteps.clear()
        self._step = 0

    # ---- private helpers -------------------------------------------------

    def _state_from_obs(self, obs: Observation) -> np.ndarray[Any, Any]:
        """Build the DT state vector from the observation dict.

        Concatenates the ``_DEFAULT_STATE_KEYS`` entries in order. The
        result must match the configured ``state_dim`` of the
        underlying DT checkpoint — mismatches surface as a
        ``ValueError`` rather than silent miscoercion.
        """
        missing = [k for k in _DEFAULT_STATE_KEYS if k not in obs]
        if missing:
            raise KeyError(
                f"DecisionTransformerPolicy.act: observation missing state key(s) "
                f"{missing}; got keys {sorted(obs.keys())}"
            )
        parts = [np.asarray(obs[k]).reshape(-1) for k in _DEFAULT_STATE_KEYS]
        state = np.concatenate(parts).astype(np.float32, copy=False)
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"DecisionTransformerPolicy.act: state dim mismatch — "
                f"checkpoint expects {self.state_dim}, observation produced {state.shape[0]}"
            )
        return state

    def _truncate_context(self) -> None:
        """Drop the oldest entries so each buffer holds at most ``context_length``."""
        max_len = self.context_length
        if len(self._states) > max_len:
            self._states = self._states[-max_len:]
            self._actions = self._actions[-max_len:]
            self._returns_to_go = self._returns_to_go[-max_len:]
            self._timesteps = self._timesteps[-max_len:]
