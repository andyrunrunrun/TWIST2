# Repository Guidelines

## Project Structure & Module Organization
`legged_gym/` contains Isaac Gym environments, task configs, and train/eval scripts (`legged_gym/legged_gym/scripts/`). `rsl_rl/` provides PPO/DAgger runners and model modules. `pose/` contains motion retargeting and `poselib` utilities/tests. `deploy_real/` holds sim2real and teleop runtime code. Top-level `*.sh` scripts (`train.sh`, `sim2sim.sh`, `sim2real.sh`, `run_motion_server.sh`) are the main entrypoints. Assets and sample motions live under `assets/`; notes/docs are in `doc/`, `docs/`, and `note/`.

## Build, Test, and Development Commands
- `conda activate twist2` then `pip install -e rsl_rl legged_gym pose`: install editable Python packages.
- `bash train.sh <exptid> cuda:0`: single-GPU student training (wrapper for `train.py`).
- `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 legged_gym/legged_gym/scripts/train.py --task g1_stu_future --proj_name g1_stu_future --exptid <exptid>`: DDP training.
- `bash eval.sh <exptid> cuda:0` or `python legged_gym/legged_gym/scripts/play.py ...`: policy playback/evaluation.
- `bash to_onnx.sh <checkpoint.pt>`: export trained policy.
- `bash run_motion_server.sh` + `bash sim2sim.sh` (or `bash sim2real.sh`): high-level + low-level deployment loop.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and PEP 8-style naming (`snake_case` for functions/variables, `CamelCase` for classes). Keep task/config names explicit (`g1_stu_future`, `g1_priv_mimic`). Prefer small, focused changes near affected modules. When changing runner behavior, keep `rsl_rl/rsl_rl/runners/on_policy_runner_mimic.py` and `rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py` aligned.

## Testing Guidelines
There is no single top-level CI script; run targeted checks before PRs:
- `python legged_gym/legged_gym/scripts/test_env.py` for environment sanity.
- `python -m pytest pose/pose/poselib/poselib/core/tests pose/pose/poselib/poselib/visualization/tests` for `poselib` unit tests.
- For policy changes, include a playback or eval run (for example, `play.py --task g1_stu_future --num_envs 1`) and report key metrics/FPS.

## Commit & Pull Request Guidelines
Recent history favors short subjects with optional scopes (`docs: ...`, `tools: ...`, `legged_gym: ...`). Prefer: `<scope>: <imperative summary>` (example: `tools: add queue-based motion eval summary`). Avoid placeholder-only messages. PRs should include:
- what changed and why,
- exact commands run for validation,
- linked issue/experiment ID,
- screenshots or logs for GUI/sim2real/teleop behavior changes.

## Security & Configuration Tips
Do not commit secrets, robot network credentials, or large private datasets. Keep machine-specific paths and interface names configurable (for example in `sim2real.sh`) and document local overrides in PR notes.
