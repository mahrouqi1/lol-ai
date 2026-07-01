# lol-ai brief

**Scope:** Per-game, per-player *win-contribution* in League of Legends via
**exact counterfactual Shapley** — players (not game-state features) as coalition
members, an on-manifold interventional replacement baseline, attributed over
win-probability increments. Bridges SHAP feature-attribution and multi-agent
credit assignment.
**Paper:** target NeurIPS/ICLR (methods) or AAAI (applied-rigorous); no deadline
set yet. No `papers/` dir yet — add paper-style imports when one is created.
**Owner:** Mazin (OSU).
**Migrated into framework:** 2026-06-07 (was a Windows-laptop repo; no prior CLAUDE.md).

## The plan that matters
The authoritative plan is the **research report in `HANDOFF.md`** (Parts C/D of
the design conversation). Read it before doing project work. Key reframe: the
*old* approach in `project_summary.md` / `agent_handoff.md` — train a win-prob
model, run TreeSHAP over **game-state features**, sum `{slot}_*` SHAP per player
— is **superseded**. That attributes to *mediators* (gold lead), not *agents*
(who created it). Those two docs are kept as historical context only.

**Immediate next step = Phase 0** (the motivating figure): take the existing
LightGBM snapshot model + `src/06_shap_explain.py`, compute SHAP twice — mean
background vs population/conditional background — and plot the attribution
disagreement. Large divergence is the empirical hook for the whole paper, and it
reuses models already trained. Do this on the workstation before building the GNN.

## Layout
- Code: `src/` (numbered pipeline: `01`harvest … `04a-d` predictors, `06` SHAP, `07` analysis).
- Data: `data/raw/` (133k matches/timelines, gitignored), `data/processed/` (parquets, gitignored, 2.2 GB).
- Models: `models/` (gitignored). Reports/figures: `reports/` (gitignored).
- Jobs: `slurm/` (OSC SLURM scripts). OSC wrapper: `osc_submit.sh`. See `OSC_WORKFLOW.md`.

## Compute policy (important)
- **Workstation = implementation/dev ONLY.** It's **shared** with other lab
  members. Use it to write code, run unit/smoke tests, iterate quickly, and make
  figures — NOT for training or long sweeps, even though it has the GPUs.
  Anything holding a GPU > a few minutes goes to OSC.
- **OSC runs all real experiments** (Phase-0 SHAP sweep, all training, the
  Shapley sweep, scale-up). See `OSC_WORKFLOW.md`.
- **Spend generously on OSC through ~2026-06-28** (funding resets then;
  substantial credit left, no rollover — use it). Still `/osc-submit-dryrun` +
  right-size every job; just don't withhold heavy runs on cost grounds until the
  reset. Revert to frugal after.

## Environment & commands (local workstation — for dev/smoke only)
- This workstation: Linux, **2× RTX 4090 (24 GB each)**, conda env
  `lol-ai` at `/research/nfs_shafieezadeh_1/mahrouqi.1/conda_envs/lol-ai` (renamed from `lol_shap_env`, moved off /home 2026-06-10).
- Note: `environment.yml` pins **CPU** torch; verify GPU torch is actually
  installed in the env (`python -c "import torch; print(torch.cuda.is_available())"`)
  before training — reinstall the cu121/cu124 wheel if it returns False.
- Run: `conda activate lol-ai && python src/<script>.py`
- Riot API key lives in `.env` (`RIOT_API_KEY=`); dev keys expire every 24 h.

## Always confirm before doing
- **OSC SLURM submission.** Run `/osc-submit-dryrun` first; state cluster,
  resources requested, est wall-clock, and est cost. Never type `sbatch` directly
  (guard hook + deny rule); use `bash osc_submit.sh`. (Cost bar is low until the
  ~2026-06-28 reset, but always still right-size and confirm.)
- **Riot ToS / deployment.** Publishing the model + repo and *per-game* lookups
  are fine (performance analysis). A **global LP-style ladder** over all ranked
  games is a prohibited "alternative skill-ranking system" — do not build or host
  one. Community-tournament leaderboards are the carve-out. A hosted tool must be
  registered with a valid API key. Don't ship a scraper (no enumerate-all-games
  endpoint; bulk harvesting breaches API terms). Flag any feature drifting toward
  a public ladder.
- Push to advisor's shared remote. Force-push. Deletion of tracked files.
- Modifications to `.github/`, `.pre-commit-config.yaml`, `slurm/*.slurm`.

## OSC settings for this project
- Account `PAS1457`; OSC user `mahrouqi1`.
- Project space: `/fs/ess/PAS1457/mahrouqi1/LoL_AI/`; big data on scratch:
  `/fs/scratch/PAS1457/mahrouqi1/LoL_AI/data/processed/`.
- Env: `pytorch/2.8.0` module + user-pip overlay
  `/fs/ess/PAS1457/mahrouqi1/envs/lol_user/` (lightgbm, xgboost, shap, seaborn,
  pyarrow, riotwatcher, python-dotenv). Build with `slurm/setup_env.sh`.
- Default cluster Ascend (A100); Cardinal (H100) heaviest single job; Pitzer for
  CPU (LightGBM CV, SHAP sweep, data processing).

## Imports (paths resolved via .framework symlink in this project)
@.framework/_docs/cross-project-learnings.md
@.framework/_lib/conventions/structure.md      <!-- always: repo layout rules (D1-D7) -->
@.framework/_lib/conventions/self-review.md    <!-- always: ultracode/autonomous self-review rule -->
@.framework/_lib/conventions/python.md
@.framework/_lib/conventions/git.md
@.framework/_lib/conventions/ml.md
@.framework/_lib/conventions/numerical-stability.md
@.framework/_lib/conventions/osc.md
@.framework/_lib/conventions/data-integrity.md
<!-- PAPER REPOS ONLY (add when papers/ exists; re-add leading @):
       .framework/_lib/conventions/latex.md
       .framework/_lib/conventions/paper-style.md
       papers/<venue>/lit-review/INDEX.md
-->

## State files
- `HANDOFF.md`  cross-chat state AND the authoritative research plan. Read at
  session start; append at end via `/memory-snapshot`.
- `OSC_WORKFLOW.md`  per-project OSC quickstart.
- Auto memory managed by Claude Code at the user level.
