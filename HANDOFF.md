# lol-shap HANDOFF log

Cross-chat state **and** the authoritative research plan. Read the latest entry
(top) before starting work. Append a new entry at session end via `/memory-snapshot`.

---

## Current state
LoL per-game player **win-contribution** project (framework-migrated 2026-06-07).
Lead model = **equivariant GNN + player-history context (04f)**; exact per-team
Shapley **contribution engine (09)** works on it. Win-pred is competitive with the
honest literature; the real targets are **early-game AUC + calibration + contribution
validity**, not pooled AUC.
- **Branch:** main. **Last touched:** 2026-06-11 (harvester crash-fix + restart).
- **Active chats:** this = PARKED monitor; **Chain-D (implementations)** owns
  `src/03*`/`src/04*`/`slurm/` (LIVE); **P2 (validation)** owns `src/09_*`+validation
  (LIVE). P1 (paper) deferred. See WORK DISPATCH PLAN below.
- **Running:** all-elo harvest (NA1+EUW1+KR, `--tiers all --players-per-tier 1000`),
  **RESTARTED 2026-06-11 13:42** (pid in `logs/harvest.pid`, log
  `logs/harvest_20260611_*.log` + `logs/bulk_harvest.log`), nohup-detached from the
  NEW project dir (`_side-projects/lol-ai`) with the NEW env
  (`conda_envs/lol-ai`; old `repos/LoL_AI` + `lol_shap_env` paths are gone).
  Previous run **died 2026-06-08** worker-by-worker (EUW1 01:46, NA1 05:55, KR 14:19)
  on unhandled `requests` transport errors (ConnectionError/ChunkedEncodingError) —
  NOT a key expiry; zero 401/403s. **FIXED 2026-06-11:** `safe_api_call` in
  `src/02_bulk_harvest.py` now catches `requests.exceptions.RequestException`
  (retry w/ 15s sleep within max_retries). ~3 days of collection lost (Jun 8–11).
  Note: seen-cache init took **~14 min** this restart (cold NFS + lab load; was 0.4s
  warm) — a freshly-restarted harvester showing only "Initializing seen caches" for
  many minutes is normal, check `ps` D-state before assuming hang.
  ⚠️ Still not crash-proof across reboots (no systemd/cron keep-alive) — re-launch
  with the same nohup command. Personal key (app limit 100/120s + 20/1s **per
  region**, confirmed from response headers).
- ✅ Canonical `models/gnn_*.pt` RESTORED (2026-06-07, Chain-D): snapshot+context from
  `models/_validated/`, and `gnn_static_model.pt` pulled fresh from OSC (patch-aware
  full-data, AUC 0.8355). Root cause fixed: 04f now honors `LOL_MODELS_DIR` (was
  hardcoded) — smokes redirect via `LOL_MODELS_DIR=models/_smoke`.

## Pending / Next
- **Chain-D:** (1) ✅ DONE **early-stopping** (`--patience`, default 3) for 04e/04f +
  04f honors `LOL_MODELS_DIR`; (2) ✅ DONE patch-indexed `03c` (per-(patch,champion)
  table) wired into `04f --static` (per-game patch lookup via `game_meta`); validated
  at scale (5506263: 100% lookups, AUC 0.8355, antisym exact, 6min vs 25min).
  **NEXT:** (3) rune/spell + **item** encoders (item needs timeline item-build
  extraction); (4) rank+patch features from `game_meta` (provide rank feature to P2
  for replacement-baseline conditioning); (5) reprocess + retrain + re-run
  `10_compare_models.py` once harvest accumulates diversity.
  - **(NEW, user-requested 2026-06-07) — ADD ABSOLUTE BLUE/RED SIDE to 04e/04f.**
    *Why:* the current models are team-RELATIVE (logit = Σ_blue score − Σ_red score,
    no absolute side feature) → hard antisymmetry `f(A,B)=1−f(B,A)` → they output
    **exactly 0.5** on any mirror matchup and **cannot represent the real ~+1.15pp
    blue-side advantage** (blue wins 51.15% in our 131,620 games; verified the model
    gives 0.500000 on mirrored states). User wants side to count **for contribution
    too**: a top laner with identical performance on blue vs red should get
    *different* contributions, because one side is inherently favored.
    *What to implement (the version that achieves the goal):* a global side-bias
    **scalar is NOT enough** — a constant cancels in each player's "real vs
    same-side replacement" marginal, so it barely moves contributions. Instead add an
    **absolute per-node side embedding** (a learned blue-vec / red-vec added to every
    node's features; all 5 nodes on a side share it → **within-team permutation
    invariance is preserved**, only the team-swap antisymmetry is dropped). This lets
    side **interact** with players via message passing, so the side advantage can
    materially change individual contributions (advantaged-side leads compound;
    strong games on the disadvantaged side count more). Keep within-team perm
    invariance EXACT (assert it); expect the model to learn ≈+1pp blue prior.
    *Downstream (already handled / notes for P2):*
      • Contribution engine `src/09` + validation suite need **no change** — "remove
        player" keeps a node on its own side and samples a **same-side, same-role**
        replacement, so side is preserved automatically and now flows into φ.
      • Validation `src/12` SYMMETRY test must be **reframed**: the mirror test
        (φ_blue=−φ_red, currently exact in fp64) will NO LONGER hold by design —
        change it to assert antisymmetry *up to the learned side prior* and report
        that residual as the measured side advantage. Efficiency + null-player tests
        are unaffected (still exact). P2 to add a NEW convergent check: blue-side
        players show systematically higher own-team contribution than red for matched
        performance, in the expected direction (~the 1pp).
      • Pair naturally with (4): a side+rank-conditioned replacement baseline.
- **P2:** ✅ C.4 validation suite BUILT + RUN (`src/11..18`, `reports/validation_report.md`):
  axiomatic/convergent/counterfactual/baseline-divergence/stability/ex-ante-expost all
  pass; predictive underpowered in apex (ties naive — needs multi-ELO). Remaining:
  re-run rank+predictive on multi-ELO; rank-conditioned baseline; OSC full sweeps;
  ΔW_t increments. New src files uncommitted. **When Chain-D ships absolute side
  (above):** reframe `src/12` symmetry to antisymmetry-up-to-side-prior, and add a
  convergent test that blue-side own-team contribution > red for matched performance.
- **Harvest:** running (see Running above); consider start-offset paging for older
  patches (timelines ~1yr). 4th continent **SEA** (+~33%) intentionally deferred per
  user — NA+EU+KR is enough for now. Reboot keep-alive (cron/systemd) **declined
  2026-06-11 per user** (workstation reboots ~1×/month); relaunch manually after
  updates.
- **Strategic / API keys (re-verified 2026-06-08 vs Riot portal):** we hold a
  **Personal** key (no expiry, 20/s + 100/2min **per region** — the per-region scope
  is confirmed both in Riot docs AND empirically by the ~83.5k/day across 3 regions;
  the old HANDOFF "GLOBAL limit" claim was WRONG). **Personal ≈ Production speed only
  via per-region parallelism**; the real lift is a **Production** key (500/10s +
  30,000/10min per region → 1M/day + Tournaments API) which is **gated on a working,
  public, player-facing product** (link to a working site/prototype, free player tier,
  ~1–3wk review). A pure research scraper does NOT qualify; a global LP ladder is
  ToS-banned. **Plan:** harvest on Personal now; once contribution results are solid,
  ship a **simple public GitHub tool** (enter a match → see per-player contribution)
  = the ToS-allowed product that earns Production. Sources in session notes below.

## Failed approaches (don't re-discover)
- 04b transformer @ lr 1e-3 (d256/6L) **diverges epoch 1** (sigmoid saturates, BCE~7,
  AUC 0.5). Fix: per-STEP warmup+cosine, grad clip 1.0, lr 5e-4.
- 04b/04c full-data **OOM** on default 8-core RAM. Fix: `#SBATCH --mem=192G`.
- `osc_submit` forwarding multi-word `TRAIN_ARGS` via `--export` **word-splits**. Fix:
  per-model slurm scripts; forward only single-token env vars.
- **04d** minute-level 3-level context: ~77min/epoch, only matches 04c → not worth it;
  **04g dropped** likewise. `AUC@end`≈1.0 is a leakage metric.
- shap **interventional TreeExplainer fails on LightGBM categorical splits** → direct
  group-Shapley via `booster.predict`.
- 04e/04f peak ep1-2 then **overfit/collapse** (val AUC 0.83→0.63) → early-stopping required.
- A `--limit` smoke writing default `LOL_MODELS_DIR` **clobbers canonical full-data models**
  → always redirect smoke output.
- Pooled AUC is misleading (late-game ≈ leakage); use early-game + calibration.
- Harvester `safe_api_call` catching only `riotwatcher.ApiError` **let `requests`
  transport errors kill worker threads silently** (one blip = one region dead,
  process exits when last worker dies; looked like key expiry but had zero 401/403s).
  Fixed 2026-06-11: catch `requests.exceptions.RequestException` + retry.
- A freshly-restarted harvester stuck on "Initializing seen caches" for many minutes
  is cold-NFS scan time (~14 min at 150k matches under lab load), not a hang.

## Recent session entries (full detail, newest at top)

### 2026-06-11 14:17  harvest-recovery (monitor chat)
**What was done:**
- Diagnosed the dead harvester: the 06-08 restart died **worker-by-worker on 06-08**
  (EUW1 01:46, NA1 05:55, KR 14:19) from unhandled `requests` transport errors
  (`ConnectionError`, `ChunkedEncodingError`) escaping `safe_api_call`, which only
  caught `riotwatcher.ApiError`. NOT a key issue — zero 401/403s in the logs.
- **Fixed** `src/02_bulk_harvest.py`: added `except requests.exceptions.RequestException`
  to `safe_api_call` (15s sleep + retry within the existing max_retries=3 budget).
- **Restarted** the harvester 13:42 nohup-detached from the new project dir with the
  new `conda_envs/lol-ai` env; verified all 3 workers downloading by 13:58.
- Updated the "Running" block above; corrected `logs/harvest.pid` to the python PID.
**What was learned:**
- ~3 days of collection lost (06-08 14:19 → 06-11 13:42); throughput was already ⅓
  for most of 06-08 as workers died one at a time.
- Seen-cache init took **~14 min** (cold NFS + heavy lab load; 0.4s warm on 06-08).
  A restart showing only "Initializing seen caches" for minutes is NOT hung —
  check for D-state on `rpc_wait_bit_killable` before assuming.
- Old launch paths are gone: `repos/LoL_AI` and `~/anaconda3/envs/lol_shap_env` no
  longer exist; launch from `_side-projects/lol-ai` with `conda_envs/lol-ai`.
- User decision: **no reboot keep-alive needed for now** (workstation reboots ~1×/mo
  for updates); manual relaunch is acceptable.
**Files modified:** HANDOFF.md (+184/-..), src/02_bulk_harvest.py (fix); the other
~32 modified files (CLAUDE.md, environment.yml, slurm/*, src/*) predate this session
(env-rename/migration churn owned by sibling chats).
**Next chat needs to know:**
- Harvester PID in `logs/harvest.pid`; logs `logs/harvest_20260611_134243.log` +
  `logs/bulk_harvest.log`. Expect ~58 games/min ≈ 83.5k/day when healthy.
- If it stops again, transport errors are now retried — check the log tail for a
  NEW failure class before re-applying the old diagnosis.
- After a workstation reboot (~monthly), relaunch manually: nohup + `--tiers all
  --players-per-tier 1000` from the project dir (see Running block).
**Backup status:** 8 untracked (src/11-18 validation suite) / 34 modified / 0
unpushed — harvest fix + HANDOFF committed this session; rest left to owning chats.
**Open questions:**
- Should the 06-08→06-11 gap shift the 250k/500k milestone ETAs communicated earlier?
- Commit P2's untracked src/11-18 validation suite (owned by P2 chat)?
**Suggested first prompt for the next chat:** "Check `tail logs/bulk_harvest.log`
and confirm the harvester is still collecting; then continue per Pending/Next."

### 2026-06-08  P2 — side-advantage decision, API-key research, harvester restart (validation chat)
**What was done (continuation of the validation session):**
- **Side advantage:** user clarified the model should represent absolute blue/red
  side AND have it flow into contributions. Verified the current GNN can't: it's
  team-relative → outputs **exactly 0.5 on mirror matchups**, while blue actually
  wins **51.15%** (131,620 games). Logged the fix as a Chain-D task (Pending):
  **per-node side embedding** (a scalar bias is insufficient — cancels in the
  same-side-replacement marginal). Memory [[lol-side-advantage-decision]].
- **API keys re-verified vs Riot portal:** Dev (24h) / Personal (no expiry, same
  limits) / Production (500/10s+30k/10min, needs public product + 1–3wk review).
  KEY FACT: app limits are **per region** (Riot docs + confirmed empirically), so the
  old "GLOBAL limit" note was wrong. Production is gated on shipping a player-facing
  tool → plan: simple public GitHub contribution tool earns it. (Sources:
  developer.riotgames.com/docs/portal; support-developer.../Production-Key-Applications;
  hextechdocs.dev/rate-limiting.)
- **Harvester:** found it had **died ~06-07 19:29** (process killed mid-backoff when
  parent shell closed; HANDOFF wrongly said "Running"). Confirmed code is correct
  (3 independent region threads NA/EU/KR, own `LolWatcher` each, per-region 429
  backoff). **Restarted nohup-detached** (`logs/harvest.pid`, `logs/harvest_*.log`);
  measured **~58 games/min ≈ 83.5k/day** → 250k in ~1.2d, 500k ~4.2d, 1M ~10d.
**For Chain-D:** (1) implement the side embedding (full spec in Pending); (2) the
harvester has no keep-alive — consider a cron/systemd or `--resume` wrapper so a
reboot doesn't silently stop data collection; SEA 4th continent deferred per user.
**For P2 (me/next):** when side lands, reframe `src/12` symmetry + add side-direction
convergent test; re-run rank-convergent + predictive once multi-ELO data accumulates.

### 2026-06-07  P2 — validation suite built + run (validation chat)
**What was done:** built the full Phase-3 contribution **validation suite**
(`src/11_validate_lib.py` shared exact estimator + `src/12..18` drivers) and ran all
seven tests on `models/_validated/` (full-data 04e snapshot 0.835/0.013 and 04f
context 0.838/0.009, re-pulled from OSC after the local-smoke clobber). Report:
**`reports/validation_report.md`** (+ `reports/validation_*.{png,json}`).
- Vectorized the estimator (all 32 coalitions in ONE forward) → ~4× speedup; folded
  into the shared lib (09 left untouched, still the production engine).
**Results (workstation-scale demos; full sweeps → OSC):**
- **Axiomatic** ✅ exact — efficiency 4.8e-17, team-swap symmetry 1.4e-16 (fp64
  mirror test; this GNN encodes role *positionally* so naive same-team symmetry
  doesn't apply), null-player 0.0.
- **Convergent** ✅ — role jungle 0.109>bottom 0.105>middle 0.091>top 0.087>utility
  0.050 (tight CIs); winners +0.063 own-team φ over losers (CI excl 0); carry/fighter
  ≈0.092 ≫ frontline/utility 0.053 (+0.039, CI excl 0). **Rank deferred** — data is
  ~all apex (141k APEX_UNTAGGED, 3.3k Chall, 30 Iron); auto-activates on multi-ELO.
- **Counterfactual** ✅ — seeded degradation: φ(real)>φ(weak)>φ(int) 100% rows,
  φ(int)<0 99.8%.
- **Predictive** ⏸ **underpowered** — apex-only, all predictors ~0.5 future-win AUC
  (matchmaking ceiling), contribution **ties** best naive proxy (winrate) Δ−0.021,
  CI [−0.089,+0.044]. Needs multi-ELO + OSC scale. Reinforces "win-pred is ceiling-
  limited" thesis. (Open Q from prev handoff: contribution does NOT yet beat naive
  stats in apex — but neither does anything; the test can't discriminate here.)
- **Baseline-divergence** ✅ — GNN-native motivating figure: top-1 flips 58–68% btwn
  baselines; on-manifold pop|cond agree most (Spearman 0.55), **mean is the outlier**
  (confirmed at K=40; K=16 was MC-noise-ambiguous).
- **Stability** ✅ — material φ rel-halfwidth 0.40, 90% sign-stable; std ∝ 1/√K
  (0.065→0.023 for K=4→32).
- **Ex-ante/ex-post** ✅ — 04f history model; seeded griefing gap +0.11→−0.22,
  separation **AUC 0.96**. Framed as behavioral-consistency, NOT detection.
**Coordination / gotchas:**
- **MODEL CLOBBER (resolved):** Chain-D's local early-stopping smoke (limit=400)
  overwrote `models/gnn_snapshot.pt`/`gnn_context_model.pt` (→0.74/0.71) mid-session.
  Re-pulled full-data copies into **`models/_validated/`** (validation-owned). The
  whole suite points there. **Chain-D action:** redirect smoke output
  (`LOL_MODELS_DIR=models/_smoke`) and `bash slurm/sync_from_osc.sh` to restore
  `models/`.
- Background runs need `PYTHONUNBUFFERED=1` and **don't pipe to `tail`/`head`**
  (the pipe buffers → output only at exit).
**Next:** (1) re-run rank-convergent + predictive once multi-ELO harvest spreads
tiers; (2) add a 4th **rank-conditioned** baseline to `src/16` when rank lands;
(3) OSC full-scale predictive + baseline sweeps; (4) re-run on ΔW_t **increments**
for the telescoping paper story. New src files are **uncommitted** (untracked).

### 2026-06-07  Chain-D — early-stopping + patch-indexed static (implementations chat)
**What was done:**
- **Task 1 — early-stopping** in 04e/04f: `--patience N` (default 3) breaks the loop
  after N epochs with no val-AUC gain. Best-val checkpoint unchanged by construction;
  antisym exact. Cancelled wasteful job 5506190 (had overfit ep1→ep26, AUC 0.839→0.63).
- **Bonus fix:** 04f hardcoded `MODELS_DIR=models/` (ignored `LOL_MODELS_DIR`, unlike
  04e) → K-sweep slurm redirect silently no-op'd AND smokes clobbered canonical models.
  Fixed to honor `LOL_MODELS_DIR`. Restored canonical models/gnn_*.pt (see Current state).
- **Task 2 — patch-indexed champion static (Phase 1.5):** 03c rewritten to emit one row
  per (patch, champion) from all 12 ddragon patches, z-scored with ONE globally-pooled
  mean/std so cross-patch drift survives (per-patch z-score would erase it). 04f `--static`
  now looks up each game's patch via `game_meta.parquet` and feeds patch-correct champion
  stats per (game, node); static feats moved out of the model (champ_static buffer/gather)
  into a per-game input tensor threaded through forward/evaluate/antisym; checkpoint stores
  static feat metadata (n_static, static_feat_cols, static_avail, static_fallback). Games on
  a patch with no ddragon dir fall back to the latest available patch.
**Result (OSC 5506263, Ascend, 6m23s, COMPLETED):** patch-static best val **AUC 0.8355**
(ep1), Brier 0.1661, **ECE 0.0245**; 1,118,770/1,118,770 + 197,430/197,430 (game,node)
champ lookups matched (100%); antisym 0.00e+00; early-stop fired ep4. vs per-champ
latest-patch static **0.8386** (5506190) vs baseline **0.8370** — all within ±0.003
single-seed noise. **Verdict: patch-indexing NEUTRAL on current 16.x-heavy data, as
predicted** (only ~5 champs vary in AD across 16.x); payoff is cross-patch generalization,
gated on harvest diversity. Both early-stopping + patch-aware paths de-risked at full scale
for the Task-6 retrain. Early-stopping cut wall-clock ~4× (6min vs 25min).
**Commits:** 8ab5df2 (early-stop), 20b1da4 (patch-index static), 9798c4c (LOL_MODELS_DIR).
**Sidecars on scratch:** new per-patch `champion_static.parquet` (2064 rows) + `game_meta.parquet`.
**Next chat (Chain-D), priority order:**
- **(NEW top priority, user-requested) absolute blue/red side in 04e/04f** — see the
  detailed spec under Pending/Next. Add a learned per-node side embedding (blue-vec/red-vec
  shared across a side's 5 nodes) so within-team perm invariance stays EXACT but team-swap
  antisymmetry is intentionally dropped; model should learn ≈+1pp blue prior and side should
  flow into contributions via message passing. Coordinate with P2 (they reframe src/12
  symmetry → antisymmetry-up-to-side-prior + add a blue>red convergent check).
- Task 3 (rune/spell static encoders — already in features; extend 03c-style table from
  runesReforged.json/summoner.json) + optional champ TEXT embeddings.
- Task 4 (item modeling, heavy — timeline ITEM_PURCHASED/SOLD/UNDO/DESTROYED inventory sim →
  per-(game,minute,slot) builds → item static encoder from item.json).
- Task 5 (rank+patch as model features — coordinate with P2: Chain-D provides rank feature,
  P2 consumes in src/09 replacement baseline; rank-convergent/predictive auto-activate then).
- Task 6 (reprocess+retrain+re-run 10_compare_models.py) gated on harvest diversity.
**Open questions:**
- Side embedding added pre- or post-message-passing? (pre lets it interact; assert perm-invariance either way.)
- Does dropping antisymmetry hurt calibration? (ablate ECE side-on vs side-off.)
- Patch-static payoff unmeasurable until harvest spans multiple metas — recheck at Task 6.
**Suggested first prompt for the next chat:** "Chain-D: implement absolute blue/red side
embedding in 04e/04f per the HANDOFF spec (learned per-side node vec, keep within-team perm
invariance exact, drop team-swap antisymmetry), smoke-verify the ≈+1pp blue prior and exact
perm-invariance, then OSC-validate."

### 2026-06-07  setup + experiments (this chat)
**What was done:**
- Migrated LoL_AI into the framework; re-grounded on the research-report plan; bootstrapped + validated OSC end-to-end.
- Built & ran on full data (OSC): Phase-0 divergence (08), equivariant GNN (04e), player-history GNN (04f), contribution engine (09); fixed 04b divergence; data-scaling study (25/50/133k); history-length K-sweep; static-context Phase 1 (00 fetch, 03c table, 03d game-meta, 04f `--static`); multi-ELO harvester (all tiers) launched; uniform comparison harness (10, accuracy + pre-game + literature).
**What was learned:**
- GNN family leads (ECE ~0.01; AUC ties LightGBM); player context helps early/pre-game (+4pp pregame acc); more data still helps (early game + calibration); K-history plateaus ~20-40 at 130k; static champ feats ~neutral now (gated on patch/elo diversity). Timelines retained ~1yr; 1M needs a production key.
**Files modified (session footprint):** src/02 (multi-elo, 266Δ), 04e/04f/08/09/10 (new), 00/03c/03d (static), 04a-d (LOL_DATA_DIR/mem/warmup), slurm/*, CLAUDE.md, OSC_WORKFLOW.md, reports/{lit_benchmark,static_context_plan,model_comparison}.
**Next chat needs to know:**
- Read the WORK DISPATCH PLAN below; P2 + Chain-D are separate live chats with disjoint files.
- All OSC training done; only the harvest runs. Early-stopping + smoke-output redirection are the #1 Chain-D fixes.
- Sidecars ready: `game_meta.parquet` (patch+tier), `champion_static.parquet` (patch-indexed), `data/raw/static/`; validated models in `models/_validated/`.
**Open questions:**
- Does Shapley contribution beat naive stats (KDA/gold/dmg share) at predicting future wins?
- Harvest-accumulation threshold before reprocessing features for rank+patch+items?
- Pursue the production key (requires shipping a public per-game tool)?
**Suggested first prompt for the next chat:** "Read HANDOFF.md; you are the validation chat — build the contribution validation suite per C.4, starting with axiomatic + convergent + predictive-vs-naive-stats on models/_validated/."

---

## 2026-06-07 — WORK DISPATCH PLAN (parallel vs series chats) + running state

**This chat is PARKED to monitor running jobs** and report results. It also owns the
data→features→models SERIES chain (below). Open new chats at `repos/LoL_AI/`.

> **⚠️ COORDINATION (2026-06-07, VALIDATION chat / P2): canonical GNN checkpoints
> got clobbered by a local smoke.** Between 19:13–19:15 today `models/gnn_snapshot.pt`
> and `models/gnn_context_model.pt` were overwritten with **limit=400 smoke models**
> (AUC 0.739 / 0.707) — almost certainly Chain-D testing the new early-stopping code
> locally with default `LOL_MODELS_DIR`. The full-data versions (0.835/0.838) survive
> on OSC; I re-pulled them into **`models/_validated/`** (validation-owned, won't be
> clobbered) and point the whole validation suite there via `--model`.
> **ASK for Chain-D:** route local smoke/dev training output away from canonical names
> — set `LOL_MODELS_DIR=models/_smoke` (the scaling study already uses this redirect)
> or never let a `--limit` run write `models/gnn_*.pt`. Then re-sync the real models
> from OSC (`bash slurm/sync_from_osc.sh`) so `models/` is trustworthy again.

### Currently running (do not duplicate)
- **All-elo harvest** (bg, days): CHALLENGER→IRON, `--tiers all --players-per-tier 1000`,
  tier-tagged → `data/raw/game_source_tier.csv`. Still early (mostly apex so far).
- **K-sweep** (5506138–42) ✅ DONE: 04f history length K=5/10/20/40/80 → val AUC
  0.833/0.835/0.837/0.837/0.838. Helps with diminishing returns; plateaus ~K=20–40
  at 130k (avg ~30 games/player); K=80 marginally best (0.838, ECE 0.009). Expect
  to keep climbing at 1M. All peak epoch 1 → add early-stopping.
- **Static Phase-1 comparison** (5506190) ✅ DONE: 04f `--static` best val AUC
  **0.8386** (ep1) vs baseline 04f **0.8370** → tiny +0.0016 (champion static feats
  ~neutral on current single-meta data, as predicted — payoff is cross-patch/cold-start,
  gated on harvest diversity). 172/172 champs matched, antisymmetry preserved. Then
  overfit hard (val AUC→0.63 by ep27). **All OSC training jobs now complete; only the
  harvest runs.**
- **EARLY-STOPPING is now a confirmed must** for 04e/04f/04f--static (they peak ep1-2
  then collapse). Chat 1's first task.
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

### 2026-06-10 orchestrator — Q-C4 post-reorg conformance sweep
**Reorg record:** renamed LoL_AI → lol-ai (moved to _side-projects/ 2026-06-09, framework-tooled); remote mahrouqi1/lol-ai added, main pushed (712 KB tracked; 139 GB data untracked). OSC ess+scratch LoL_AI → lol-ai (correction: this repo DID have OSC workspaces); 9 slurm/sync files fixed; CLAUDE.md header lol-shap → lol-ai.
**Conformance:** .framework + imports OK; environment.yml present; zero stale names/abs paths.

### 2026-06-11 orchestrator — workstation conda env: how to use + extend
**Env:** `conda activate lol-ai` (by name, never absolute path). Lives on NFS at `conda_envs/lol-ai`; the repo-root `environment.yml` (curated spec) is the source of truth, the env on disk is disposable. Single source for everything env-related: `_framework/_docs/conda-environments.md` — read it before changing the env. This repo's env: torch 2.5.1+cu121, curated spec — pip block needs the pytorch extra-index line (see doc) if ever re-solved.
**Installing new packages — do it this way:**
- First, the standard exports (NFS=`/research/nfs_shafieezadeh_1/mahrouqi.1`): `export TMPDIR=$NFS/conda_envs/.build/tmp PIP_CACHE_DIR=$NFS/conda_envs/.build/pip-cache PYTHONNOUSERSITE=1`. PYTHONNOUSERSITE is new (2026-06-11): a stale RHEL-patched pip in `~/.local/lib/python3.11/site-packages` shadows every py3.11 env and breaks ALL in-env pip runs on the Ubuntu node (196872) with a "TLS CA certificate bundle" error; same command works on the other node, so it looks node-random.
- conda deps: `$NFS/conda_envs/.tools/bin/mamba install -p $NFS/conda_envs/lol-ai <pkg>` — never the base classic solver (it spins for hours).
- pip deps: activate the env, then `python -m pip install ...` (with the exports above). torch / PyG wheels need their extra index URLs — copy from the doc's pip-pitfalls section.
- After any material change, hand-update the curated `environment.yml` (science pins at major.minor; never paste a frozen export — they rot). Optionally refresh `environment.lock.yml` via `conda env export --no-builds` as a forensic record only.
**OSC is separate** — never point OSC slurm scripts at this env name, and don't "fix" workstation scripts to use OSC module names.
