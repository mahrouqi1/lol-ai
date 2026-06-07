# lol-shap HANDOFF log

Cross-chat state **and** the authoritative research plan. Read the latest entry
(top) before starting work. Append a new entry at session end via `/memory-snapshot`.

---

## 2026-06-07 — WORK DISPATCH PLAN (parallel vs series chats) + running state

**This chat is PARKED to monitor running jobs** and report results. It also owns the
data→features→models SERIES chain (below). Open new chats at `repos/LoL_AI/`.

### Currently running (do not duplicate)
- **All-elo harvest** (bg, days): CHALLENGER→IRON, `--tiers all --players-per-tier 1000`,
  tier-tagged → `data/raw/game_source_tier.csv`. Still early (mostly apex so far).
- **K-sweep** (5506138–42) ✅ DONE: 04f history length K=5/10/20/40/80 → val AUC
  0.833/0.835/0.837/0.837/0.838. Helps with diminishing returns; plateaus ~K=20–40
  at 130k (avg ~30 games/player); K=80 marginally best (0.838, ECE 0.009). Expect
  to keep climbing at 1M. All peak epoch 1 → add early-stopping.
- **Static Phase-1 comparison** (OSC job 5506190): 04f `--static` vs baseline 04f
  (AUC 0.837); 172/172 champs matched; result pending (will land in HANDOFF/next chat).
- 04d cancelled; 04g dropped (minute-history ≈ game-history at huge cost).

### PARALLELIZABLE — independent, spin up as SEPARATE chats (disjoint files)
- **Chat P1 — Paper & related-work** (files: `papers/` only). Start drafting from the
  research plan (this HANDOFF) + `reports/lit_benchmark.md` + `reports/static_context_plan.md`.
  Fully independent of code.
- **Chat P2 — Contribution & validation suite** (files: `src/09_*`, new validation scripts,
  `reports/`). Uses EXISTING checkpoints (`models/gnn_snapshot.pt`, `gnn_context_model.pt`)
  → build C.4 validation (predictive / convergent / counterfactual / axiomatic), the
  ex-ante/ex-post gap, seeded-griefing case study. Does NOT touch `src/03*`/`src/04*`.

### SERIES — one chat, strictly ordered (shares features.parquet + src/04*; do NOT parallelize)
- **Chain D — data → features → models:**
  1. Let harvest accumulate multi-elo + patch diversity.
  2. Reprocess features adding **rank** (`game_meta.source_tier`) + **patch** (`game_meta.patch`)
     + **item features** (item-build extraction from timelines, folded into `03`/`03b`).
  3. Retrain 04a–04f on enriched/larger data.
  4. **Patch-index** the champion static encoder (per-game patch via `game_meta`) +
     **rank-condition** the replacement baseline.
  5. Re-run `src/10_compare_models.py` (comparison + scaling) and `src/09` (contribution).
  Each step depends on the prior — keep in one chat.

### Coordination rules
- File ownership: P1=`papers/`, P2=`src/09_*`+validation, Chain-D=`src/03*`/`src/04*`/`slurm/`.
  Avoid cross-editing to prevent conflicts.
- OSC only via `bash osc_submit.sh` (after `/osc-submit-dryrun` + confirm). Commit before submit.
- `HANDOFF.md` is the shared state — read latest before starting; append via `/memory-snapshot`.
- Sidecars ready for Chain D: `data/processed/game_meta.parquet` (patch+tier),
  `data/processed/champion_static.parquet`, `data/raw/static/<patch>/` (ddragon).

---

## 2026-06-07 (latest) — Phase 2 contribution engine works on trained 04e GNN

[src/09_contribution_gnn.py](src/09_contribution_gnn.py): exact per-team
32-coalition Shapley on the trained equivariant GNN, in win-PROB space. "Remove
player" = swap node features to a ROLE-CONDITIONED on-manifold real replacement;
interactions carried by message passing; other team fixed. Reuses 04e via importlib.
- **Efficiency residual 5.2e-17** (exact — Shapley sums to v(full)-v(empty)).
- **Convergent validity:** mean |contribution| by role = bottom 0.123 > jungle
  0.108 > middle 0.094 > top 0.089 > utility 0.051 (carries swing win-prob most,
  support least — matches LoL domain knowledge). Figure: reports/gnn_contribution_example.png.
This is the paper's core method working end-to-end on a real, calibrated model.

**DATA-SCALING STUDY (2026-06-07):** to decide if harvesting more data (toward
1M games) is worth it, reran 04b/04c/04e at 25k and 50k games (full=133k done).
Infra: `LIMIT` env (single-token, forwarded via osc_submit) + `LOL_MODELS_DIR`
redirect to `models_scale/<N>/` so full-data models aren't clobbered.
Jobs (Ascend): 25k=5505488(04b)/5505489(04c)/5505490(04e);
50k=5505491(04b)/5505492(04c)/5505493(04e). Read learning curve from job logs
(val AUC/Brier/ECE per N) → if curves are flat by 133k, more data isn't worth it.

**Full-data 04f ✅** AUC **0.837** (best so far), ECE 0.025, antisymmetry exact.
Saved gnn_context_model.pt.

**Background poller running** (task watches Ascend squeue every 3min, notifies on
completion). Then: pull all models/logs → comparison harness (recompute
AUC-by-minute + calibration on common held-out set) + scaling curve plot.

**04g (minute-context GNN):** still gated — decide after 04d result + scaling.

**04b/04c first full runs OOM-killed; fixed with --mem (cross-project lesson).**

**04b transformer was BROKEN (diverged: AUC 0.5, BCE ~7) at every scale** — fixed
2026-06-07: per-step LR warmup+cosine, grad clip 5.0->1.0, peak lr 5e-4. Smoke
confirms learning (AUC by min 0.54->0.88). Rerunning: full=5505526, 25k=5505527,
50k=5505528 (broken 50k 5505491 cancelled).

**HARVESTER AUDIT (02_bulk_harvest.py) + DATA-SCALING FEASIBILITY:**
- Throughput is API-rate-limited, not code-limited. **Dev key ≈ 100 req/2min
  GLOBAL** (multi-region threading does NOT bypass the app-wide limit), ~2 req/game
  -> **~25 games/min ≈ 36k/day** continuous; dev keys expire every 24h (needs
  babysitting). 133k set took ~17 calendar days intermittent.
- **1M games: impractical on a dev key (~a month of babysat harvesting).**
- **API KEY TYPES (researched 2026-06-07):**
  - *Development:* 20 req/s, 100 req/2min, **expires every 24h** (current).
  - *Personal:* **SAME limits** (20/s, 100/2min — "won't be approved for rate
    increases") but **no expiry**; explicitly allows research/private use; cannot
    be public-facing. → Removes the 24h babysitting but does NOT speed up harvest.
    Realistic unattended: ~36k/day → ~250k in ~7d, ~500k in ~14d.
  - *Production:* **500 req/10s, 30,000 req/10min** (~36-60x faster → 1M in <1 day),
    + Tournaments API. BUT requires a **working, PUBLIC-FACING product that benefits
    players**, reviewed for quality/completeness (~1-3 week review).
  - **CATCH:** a pure research harvester does NOT qualify for Production (not a
    player product). To get Production we'd have to ship a real player-facing tool
    — e.g., the per-game contribution analyzer (ToS-ALLOWED) — NOT a global ladder
    (ToS-BANNED, see Part A). So: Personal = easy, removes babysitting, same speed;
    Production = fast but is a product-building + review commitment.
  - **Recommendation:** Personal key now for unattended ~250-500k; pursue Production
    only if/when we build the per-game contribution tool as a shippable product.
- **Patch coverage:** current 133k already spans **6 patches** (15.24, 16.1-16.5).
  No `patch` column in features yet — `gameVersion` is in raw match JSON. To use
  patch as a signal we must extract it (info.gameVersion) into features.

**FUTURE WORK — patch/static-context encoder (user request, keep in mind):** give
the model a "game context" = per-PATCH static stats of all champions, items, runes,
objectives/epic monsters, minions. Implement as token encoders (champ-token,
item-token, rune-token, ...), ideally fusing TEXT (names/descriptions) + NUMBERS
(base stats/scalings) per entity. Rationale: across many patches the meta shifts
(champ/item numbers change), so a patch-conditioned static context lets one model
generalize across patches. **Data NOT yet downloaded/processed** — source is Riot
Data Dragon / CommunityDragon (free, no rate limit, versioned per patch:
ddragon/cdn/<version>/data/.../champion.json, item.json, runesReforged.json).
Substantial build; revisit when scaling data across many patches. See memory
[[patch-static-context-encoder]].

**PHASE 1 IMPLEMENTED (2026-06-07):** `src/00_fetch_static_data.py` (ddragon, all 12
patches → data/raw/static/); `src/03c_build_static_features.py` → champion_static.parquet
(24 numeric + 6 class tags + 13 partype). `04f --static` feeds patch-aware champion
static features per node (→ gnn_static_model.pt, backward-compatible); antisymmetry
preserved, 167/167 champs matched. Full comparison vs baseline 04f (AUC 0.837):
job **5506190**. v1 = latest-patch stats (patch-agnostic; cross-patch payoff needs
patch-indexing + multi-patch data). Next: rune/spell encoders; item encoder (needs
timeline item-build extraction); monsters/objectives (curated table). Design doc:
reports/static_context_plan.md.

**MULTI-ELO HARVESTING — DONE + RUNNING (2026-06-07).** 02_bulk_harvest.py now
supports ALL tiers CHALLENGER→IRON (apex via league endpoints incl MASTER;
standard via paginated league.entries; all return puuid). `--tiers all
--players-per-tier 1000` launched (3 regions, balanced ~1000 players/tier). Each
game tagged with its source tier in `data/raw/game_source_tier.csv` (approx game
elo → future rank feature). NOTE: the pre-existing ~141k apex games are NOT in the
tag file (treat untagged = Challenger/GM). KEY SYNERGY: the replacement baseline
conditions on (rank, role, patch); rank was ~constant (all apex) so that was moot
— multi-ELO makes rank-conditioning real and strengthens the contribution framing.
Next: after harvest, extract rank (from tag file) + patch (from gameVersion) into
features; rerun processing pipeline. See [[lol-multi-elo-and-ssl]].

**TODO (history-richness feature, post K-sweep):** 04f's HistoryEncoderGame uses a
masked MEAN over valid games → it normalizes away HOW MANY games a player has.
Add an explicit richness signal (count / log-count of valid history games, and/or
fraction-of-K-filled) into the node embedding. Rationale (user insight): history
availability is itself signal — confidence of the skill estimate + smurf/new-account
vs veteran. Also the principled cold-start handle (thin history → lean on bucket
prior). The current K-sweep is the count-agnostic baseline; rerun with richness to
measure the delta. Expected to matter more at high K and much more at 1M scale.

**FUTURE WORK — self-supervised pretrain -> finetune (user request):** SSL on the
abundant per-minute multi-agent series + player histories (masked-feature/next-
state modeling, contrastive same-player-across-games, masked champ/item tokens),
pretraining the encoders (player-history encoder, GNN node encoder, patch/static
encoders); then finetune the win-prob head on outcomes. RATIONALE: supervised win
labels hit the ~50/50 structural ceiling (limited signal), but the minute-by-minute
data is rich and unlabeled — SSL should yield better representations (esp. early
game + calibration + contribution quality) than outcome-only training. Natural
backbones to pretrain = the history encoder + the GNN. See [[lol-multi-elo-and-ssl]].

---

## 2026-06-07 — Phase 1 GNN built + OSC fully bootstrapped

**Phase 1 predictor** [src/04e_train_gnn.py](src/04e_train_gnn.py): equivariant
per-minute GNN (10-node match graph). Hard symmetries verified EXACT:
within-team permutation invariance + team-swap antisymmetry (residual 0.00e+00).
Composes with the contribution engine (remove player = swap node). Reads
`LOL_DATA_DIR`. Smoke val AUC ~0.77-0.82 (overfits at tiny scale; needs full data).
NOTE: model is small (217k params) and **data-movement-bound** — GPU util ~0% on
small runs. Before big sweeps, consider pre-tensorizing to .npz on scratch; a
single A100 is fine but the GNN itself is cheap.

**OSC is fully bootstrapped and validated** (key-auth + multiplexing → only in-chat
approval needed, no password/Duo):
- Dirs created: `/fs/ess/PAS1457/mahrouqi1/LoL_AI/` + `/fs/scratch/PAS1457/mahrouqi1/LoL_AI/data/processed/`.
- Code synced; `features.parquet` (2.2 GB) pushed to scratch (~100 MB/s).
- Env overlay built (`slurm/setup_env.sh`): pytorch/2.8.0 + cuda/12.8.1 modules
  exist on Ascend; overlay `lol_user` has lightgbm/xgboost/shap/seaborn/etc.
- **Smoke job 5505394 on Ascend succeeded end-to-end** (~$0.03): A100, CUDA True,
  data read from scratch, GNN trained 3 epochs, antisymmetry exact, model saved.
- Blessed submit path works: `bash osc_submit.sh slurm/<job> <cluster>` (raw
  `sbatch` stays deny-listed). `slurm/smoke_test.slurm` + `slurm/train_gpu.slurm`
  now default to the GNN; train_gpu honors `TRAIN_SCRIPT`/`TRAIN_ARGS`.
- Budget: ~0 used so far this period (plenty; resets ~06-28).

**DONE since:** patched 04a-04d for `LOL_DATA_DIR`; pushed `player_game_summary`
to scratch; added per-model slurm scripts (train_cpu, train_04b, train_04c) after
hitting a `sbatch --export` multi-word word-split bug (fixed osc_submit to forward
single-token vars only).

**FULL-DATA TRAINING RESULTS/STATUS (2026-06-07):**
| Model | Job | Status | Result |
|------|-----|--------|--------|
| 04a snapshot LightGBM | 48074894 (Pitzer) | ✅ DONE | AUC by min: 5-10=0.75, 10-15=0.83, 15-20=0.87, 20-25=0.90, 25+=0.90. Fresh 478-feat model. |
| 04e equivariant GNN | 5505414 (Ascend) | ✅ DONE | val AUC **0.834**, Brier 0.166, **ECE 0.013** (excellent calib), antisymmetry 0.0. ~12min. |
| 04f GNN + game context | 5505445 (Ascend) | ⏳ running | the model the user requested; see below |
| 04b causal Transformer | 5505405→**5505451** | ♻ resubmitted | first run OOM-killed; +`--mem=192G` |
| 04c player-context Transf | 5505406→**5505452** | ♻ resubmitted | first run OOM-killed; +`--mem=192G` |
| 04d minute-context | **5505474** (Ascend) | ▶ running | sequences built (33.6M rows, 446MB) + pushed to scratch |

**LESSON (candidate cross-project learning):** OSC GPU jobs with big-data
preprocessing (StandardScaler/sequence-padding on millions of rows) OOM on the
default 8-core RAM share. Add explicit `#SBATCH --mem=192G` (Ascend has 1 TB;
memory isn't the cost driver, GPU-hours are). Done for 04b/04c/04d/04f.

**Early read:** the **equivariant GNN (04e) is the standout** — it ties the
LightGBM AUC ceiling but with dramatically better calibration (ECE 0.013), which
is exactly what the contribution method needs. And it's the model the 32-coalition
engine plugs into natively.

**04d pipeline (in progress):** 03b builds sequences from RAW JSONs (`data/raw/`,
local only — NOT on OSC), so `player_minute_sequences.parquet` is being built on
the workstation (`03b --include-sequences --workers 8`, ~8 GB out). Then: push to
scratch → `bash osc_submit.sh slurm/train_04d.slurm ascend`. (If we later want to
re-process/scale data on OSC, raw must be pushed there first — small-file transfer
is slow; tar+push is the better route.)

**Next:** monitor jobs → `sync_from_osc.sh` to pull models/reports → compare
(AUC-by-minute, calibration ECE/Brier, early-game discrimination). Then run the
contribution engine (08) on the best/most-relevant trained model; build 04e full
run; consider 04d sequences + more data harvesting.

---

## 2026-06-07 (later) — Phase 0 implemented + first result

**Built** [src/08_phase0_baseline_divergence.py](src/08_phase0_baseline_divergence.py):
holds the LightGBM snapshot model fixed, varies only the replacement baseline,
attributes at the **player level** via exact per-team interventional Shapley
(2^5=32 coalitions/team) computed directly with `booster.predict` + background
swapping. This sidesteps a real blocker — **shap's interventional `TreeExplainer`
cannot handle this model's LightGBM categorical splits** (`TreeEnsemble has no
attribute values`); the direct group-Shapley does, and it's exactly the Phase-2
estimator previewed at slot level.

**Model/data caveat found:** `models/lgbm_snapshot.txt` (Mar 10) was trained on
**401** features; current `features.parquet` has 478. Script drives the feature
list from `booster.feature_name()` so it explains exactly what the model expects.
Available conditioning columns: `region`, `minute` only (no patch/rank — all
Challenger+GM). GPU torch IS installed (2.5.1+cu121) despite env.yml's CPU pin.

**FIRMED RESULT (200 games, 5,082 rows, K=24, pool 77.5k rows; 109s on workstation).**
Metric refinements added: decisiveness-weighted flip-rate (down-weights ambiguous
near-zero rows), per-game integrated attribution, minute-bucket breakdown.
- **mean-bg vs cond-bg** (headline): top contributor flips **47.4%** per-minute
  raw, **32.0%** decisiveness-weighted; **46.5%** of games per-game-integrated;
  Spearman 0.64 / 0.66.
- **Structure of disagreement** (the paper story): the off-manifold **mean** baseline
  is the outlier — it disagrees with everything (43-51%). The two on-manifold
  baselines (pop, cond) AGREE with each other (29% flip, Spearman 0.87-0.92) but
  differ from mean AND from the legacy tree-path method (35% per-min / 24% per-game).
- **Not an early-game artifact:** flip rate is 44-53% across ALL minute buckets and
  RISES into late game (25+: 53%). Decisiveness-weighting still leaves ~1/3 of
  *confident* attributions flipping.
Figures: `reports/phase0_baseline_divergence.png` (scatter + flip bars),
`reports/phase0_pairwise_disagreement.png` (Spearman/flip/L1 heatmaps).
**Verdict: Phase 0 succeeds — the baseline choice materially changes "who was the
best player," and the off-manifold mean baseline is the worst offender.** This is
the empirical hook justifying the population-conditional replacement baseline.

**Open / next:**
- Optional: full 133k-game sweep → OSC Pitzer CPU (needs one-time OSC bootstrap:
  setup_env + push parquet & model to scratch). Numbers above are already solid;
  full sweep is for the final paper figure. Funding generous through ~06-28.
- Phase 1: build the equivariant temporal GNN predictor (the real model).
- Conceptual caveat to keep stating: Phase 0 is on the game-state model
  (mediator-level) by design — it's the motivator, not the final method. Also
  retrain 04a on the current 478-feature parquet at some point (model is stale).

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

### Compute policy: workstation = implementation only; OSC = all real runs
- **Workstation (2× 4090) is SHARED** with other lab members → use it for
  **implementation/dev only**: writing code, unit/smoke tests, quick iteration,
  inspecting outputs, figures. Do NOT tie it up with training or long sweeps,
  even though it's capable. Anything that holds a GPU for more than a few minutes
  belongs on OSC.
- **OSC (see `OSC_WORKFLOW.md`) runs all actual experiments:** Phase-0's full
  SHAP sweep, every training run (transformers now, the GNN later), the Phase-2
  Shapley sweep (parallel single-GPU jobs), and 1M-game scale-up (Pitzer hugemem).
- **Spend generously through ~2026-06-28.** OSC funding resets then; substantial
  credit remains and does NOT roll over — favor running heavy/parallel work on
  OSC *now* rather than deferring. Cost guardrails still apply mechanically
  (`/osc-submit-dryrun`, right-size, no `--exclusive`, no blind 4-GPU), but the
  "is this run worth it?" bar is low until the reset. After the reset, revert to
  frugal defaults.

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
**Legacy promotion:** COMPLETE (2026-06-07). No `CLAUDE.legacy.md` (case-3
migration). Sole user-level auto-memory entry (`lol-ai-migrated-7th-project`)
is orchestrator-tracking context and correctly stays in auto-memory — nothing
elevated to CLAUDE.md or conventions.
**Open:** decide whether to keep `settings.local.json` (24 Windows-era lines).
