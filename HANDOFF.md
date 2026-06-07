# lol-shap HANDOFF log

Cross-chat state **and** the authoritative research plan. Read the latest entry
(top) before starting work. Append a new entry at session end via `/memory-snapshot`.

---

## 2026-06-07 — Re-grounding + framework setup (orchestrator session)

This session migrated LoL_AI into the framework and re-grounded the project on
the **research-report plan** (below). The old `project_summary.md` and
`agent_handoff.md` describe a *superseded* approach — kept only as history.

### Where the project actually is

**Data (on the workstation, gitignored):**
- `data/raw/`: 133,431 matches · 120,435 timelines · 43,456 player-mastery JSONs.
  NA1/EUW1/KR, Challenger+GM, ranked solo (queue 420), patches 16.2–16.4.
- `data/processed/features.parquet` — 2.2 GB, ~3.59M rows × 482 cols (per game×minute).
- `data/processed/player_game_summary.parquet` — 72 MB, 1.33M rows, ~43k players.
- Old SHAP outputs present (`shap_*.parquet`) — these are the *game-state* SHAP
  (the superseded approach); useful only as Phase-0 raw material.

**Models trained (gitignored, `models/`):**
- `lgbm_snapshot` (LightGBM, OOF AUC ~0.816), `lstm_timeseries`,
  `transformer_timeseries` (~0.814), `player_context_model`,
  `player_context_minute_model`. All from Feb–Mar 2026.

**Compute:** workstation = 2× RTX 4090 (24 GB); conda env `lol_shap_env` exists.
OSC ready via `slurm/` + `osc_submit.sh` + `OSC_WORKFLOW.md` (account PAS1457).

### THE PLAN (research report, condensed — this supersedes the old docs)

**Reframe (most important):**
1. **Win-prediction accuracy is NOT the objective.** ~70–75% pregame in
   Challenger/GM is the structural ceiling (matchmaking → 50/50). High *in-game*
   accuracy = leakage (reading the scoreboard). Optimize **calibration**
   (ECE/Brier) + **intervention deltas**, not accuracy.
2. **The characteristic function `v(S)` is the whole paper.** The real question
   is *Shapley over what game?* — what `v(S)` means and where the counterfactual
   comes from when a player is removed.
3. **The replacement-player baseline IS the modeling contribution.** Baseline =
   an **expectation over outcomes**, not a point in feature/embedding space.
   Average *predictions*, never representations (mean-of-features and
   mean-of-embeddings are both off-manifold).
4. **Principled baseline (resolves "the default player"):** interventional /
   marginal expectation by sampling **real** player-histories from the population
   conditioned on (rank, role, patch[, champion]) and averaging the frozen
   model's win-prob output:
   `φ_baseline(i) = E_{h ~ P(history | rank, role, [champion])} [ f(history_i ← h) ]`.
   On-manifold by construction.
5. **Interventional, not conditional** (Janzing 2020): "this player added X,
   others held fixed," no credit leaking via teammate correlations. Mention
   conditional, justify in two sentences.
6. **Cause vs mediator:** game state at minute *t* is a **mediator** of player
   actions. Naive game-state SHAP attributes to mediators (the gold lead), not
   agents (who made it). Intervene on the **player**, let state roll forward.
7. **Structure & symmetry:** 10 nodes, 2 teams, lane-matchup edges. Bake in
   within-team permutation invariance + team-swap antisymmetry `f(A,B)=1−f(B,A)`.
   Exact Shapley over 5 teammates = **32 coalitions** → exact attributions.

**Temporal + aggregate (resolved):** attribute win-prob **increments**
`ΔW_t = W_{t+1} − W_t` per window via exact 32-coalition Shapley with the
replacement baseline; sum over t for the overall (linearity → per-window sums to
overall; `Σ ΔW_t = W_T − W_0`, telescoping). Increments avoid re-crediting a
standing lead each window. Scope = **realized ex-post contribution** (VAEP
family), NOT a counterfactual re-simulation (no world model) — state this.

**Headline decomposition:** *ex-ante* (swap whole history → predict final WP from
identity/skill = expected contribution) vs *ex-post* (trajectory-integrated =
actual). **Gap = over/under-performance vs the player's own expectation** — the
right signal for the griefing/intent case study (never the headline result).

**Champion-conditioning fork:** champion-conditioned replacement isolates *pilot
skill* (recommended headline); champion-agnostic bundles champion *choice*.
Different metrics — pick deliberately (leaning conditioned).

### Phased roadmap

- **Phase 0 — one-figure motivator (DO FIRST, on workstation):** existing
  LightGBM snapshot + `src/06_shap_explain.py`; compute SHAP with (a) mean
  background vs (b) conditional/population background; plot the disagreement.
  Cheap, reuses trained models. Large divergence justifies the whole project.
- **Phase 1 — predictor:** equivariant temporal GNN over the 10-player
  interaction graph with a player-history node encoder; track **calibration**;
  verify exact antisymmetry. (Existing `04b/04c/04d` are precursors, not the
  final model.)
- **Phase 2 — exact contribution:** 32-coalition Shapley on increments with the
  champion-conditioned replacement; check efficiency (`Σφ ≈ W_T − W_0`). Cost
  ≈ `32 × N × T × 10` forward passes/game — batchable, parallelizable → OSC.
- **Phase 3 — validation suite** (the suite *is* a contribution; no ground truth):
  predictive (high-contrib players win more in held-out future games),
  convergent (vs rank/known carries/pro-vs-amateur), counterfactual (seeded
  synthetic inting / skill-mismatched slots), axiomatic (efficiency/symmetry/
  null), and the **baseline-divergence ablation** (= Phase 0, marginal vs
  conditional background).
- **Phase 4 — application:** ex-post − ex-ante deviation as latent
  intent/behavioral-consistency case study. **No public gameplay griefing
  labels exist** — frame as behavioral consistency / deviation, not "detection."

### Workstation vs OSC split
- **Workstation (2× 4090):** Phase 0, prototyping, LightGBM, current-scale
  (133k) transformer/GNN training, debugging, figures.
- **OSC (see `OSC_WORKFLOW.md`):** the heavy GNN once it outgrows a 4090; the
  Phase-2 Shapley sweep over many games (parallel single-GPU jobs); 1M-game data
  scale-up (Pitzer hugemem). Always `/osc-submit-dryrun` + confirm cost first.

### Riot ToS (deployment constraints — keep in view)
Publishing model + repo = fine. Per-game lookup = fine (performance analysis).
**Global LP-alternative ladder = prohibited.** Community-tournament leaderboard =
allowed carve-out. Hosted tool must be registered + keyed. No enumerate-all-games
endpoint; don't ship a scraper. Risk follows the operator. (Full analysis: Part A
of the design conversation; summary lives in CLAUDE.md.)

### Open questions
1. Champion-conditioned vs agnostic as the headline metric (leaning conditioned).
2. Window definition: fixed-time (clean telescoping) vs event-based per
   teamfight/objective (interpretable). Likely time-based math + event overlays.
3. Node encoder: history length (20? variable?), cold-start fallback to bucket
   prior for thin histories.
4. CIs: bootstrap over replacement samples.
5. Team-swap antisymmetry as hard constraint vs soft penalty — ablate calibration.
6. Bucket granularity for the prior: bias/variance; hierarchical pooling across patches.

### Onboarding checklist for the next (per-project) chat
Open a Claude Code chat **at `repos/LoL_AI/`** so it loads THIS project's
CLAUDE.md (not the orchestrator brief). Then:
1. Read CLAUDE.md + this HANDOFF entry. Skim `project_summary.md` /
   `agent_handoff.md` ONLY for data/script facts — their *method* is superseded.
2. `conda activate lol_shap_env` then
   `python -c "import torch; print(torch.cuda.is_available())"`. If False,
   reinstall a CUDA torch wheel (env.yml pins CPU torch).
3. Verify data: `python -c "import pandas as pd; print(pd.read_parquet('data/processed/features.parquet', columns=['game_id']).game_id.nunique())"`.
4. Run `/promote-legacy` (required post-migration step) to triage any leftover
   legacy/auto-memory content.
5. Start **Phase 0**: adapt `src/06_shap_explain.py` to compute + compare
   mean-background vs population-background SHAP; produce the disagreement figure.

---

### 2026-06-07 migration → framework v0.6.8
**What was done:** ported existing project into framework v0.6.8 (no prior CLAUDE.md).
- `.claude/settings.json` replaced (was stale Windows allow-rules) with the
  framework moderate template (workstation hostname substituted).
- `.claude/hooks/` installed (4 hooks); 14 skills symlinked; `.framework` symlink added.
- `.claude/settings.local.json` (24 lines, Windows-era) preserved untouched —
  review whether any rules are worth promoting, else delete.
- git initialized; baseline commit + framework commit on branch
  `framework-migration-2026-06-07`.
**Open:** run `/promote-legacy`; decide whether to keep `settings.local.json`.
