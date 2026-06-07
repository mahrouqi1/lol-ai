# Win-prediction: ours vs literature (related-work baseline)

OURS — apex (Challenger/GM) only, leakage-free, accuracy @0.5:

```
          our comparable result value (apex, leak-free)
     pre-game: draft only (04e)   acc 0.524 / auc 0.536
pre-game: draft + history (04f)   acc 0.563 / auc 0.591
  first 10 min (04f early 0-10)   acc 0.652 / auc 0.716
            mid 15-20 min (04f)   acc 0.790 / auc 0.876
             late 25+ min (04e)   acc 0.823 / auc 0.910
       pooled all-minutes (04f)   acc 0.747 / auc 0.837
```

LITERATURE reference baselines:

```
 game                              stage metric      value       elo  leak                                                   source
  LoL               pre-game: draft only    acc  0.55-0.57 mixed/pro False                              TechLabs Aachen; LoLDraftAI
  LoL  pre-game: draft + player winrates    acc 0.879-0.90     mixed  True                             LeagueOfPredictions (GitHub)
  LoL                       first 10 min    acc 0.73-0.744     mixed False                  ML Methods for LoL Outcome; IEEE CoG'21
  LoL     intermediate (~60-80% elapsed)    acc      0.816     mixed False                    ML Methods for LoL Outcome (LightGBM)
  LoL              pro, late / full game    auc       0.95       pro False                LoL Real-Time Result Prediction (XGBoost)
  LoL           pre-game w/ player stats    auc       0.97     mixed  True                                    RF/LR on player stats
Dota2                          full game    acc  0.88-0.89     mixed False                          arXiv 2106.01782 (GBM; NN/LSTM)
Dota2                    >20 min matches    acc      0.986     mixed  True                   ExtraTrees + hero/item emb (late-game)
Dota2                         draft only    acc      ~0.70     mixed False                                          Semenov & Romov
  LoL player action scoring (no win-AUC)      -          -         - False Action2Score, arXiv 2207.10297 (GRU; not counterfactual)
```

## Caveats for fair comparison
1. AUC != accuracy: many papers report accuracy; compare like-for-like.
2. Elo regime: ours is apex-only (matchmaking ~50/50 -> harder); most papers use
   mixed/low elo or pro (bigger skill gaps -> easier -> higher numbers).
3. leak=True rows are inflated by historical win-rate features or late-game
   (>20min) snapshots; we deliberately avoid these and report calibration (ECE).
4. Most literature stops at win-prediction; our contribution method (exact
   per-team Shapley + on-manifold replacement + equivariant GNN) is the novelty.
