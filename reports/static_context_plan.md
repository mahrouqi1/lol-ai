# Patch / static-context encoder — scoping & design

**Goal:** give the models a *patch-aware* "game context" — the static properties of
champions, items, runes, summoner spells, and (later) monsters/minions/objectives
for the game's patch. So one model generalizes across patches (when a champ/item is
buffed/nerfed, the encoder sees the new numbers) and across new entities, instead
of memorizing per-ID embeddings tied to one meta. Matters most as the dataset spans
many patches (our data already covers 15.24 + 16.1–16.11; the ongoing harvest adds more).

## Data sources & status

| Source | Entities | Fields | Status |
|---|---|---|---|
| **Data Dragon** (free, versioned, no key) | champions, items, runes, summoner spells | numbers + categories + text | ✅ downloaded for all 12 patches (`src/00_fetch_static_data.py` → `data/raw/static/<ver>/`) |
| **CommunityDragon** / curated table | monsters (baron/dragons/herald/grubs/camps), minions, turrets/objectives | gold, hp, stats | ❌ TODO — not in ddragon; ~15–20 entities, change slowly → small curated per-patch table is simplest |

**Fields available (ddragon):**
- **Champions (172):** ~24 numeric base stats + per-level growth (hp, armor, ad, as, ms, range, regen, …); `info` (attack/defense/magic/difficulty 0–10); `tags` (Fighter/Tank/Mage/Assassin/Marksman/Support); `partype` (resource); **text** (name, title, blurb, lore, 4 spell tooltips, passive).
- **Items (705):** `gold` (base/total/sell/purchasable); structured `stats` stat-mods; `tags` (Damage/CriticalStrike/…); **text** (name, description, plaintext).
- **Runes (5 trees):** name/shortDesc/longDesc **text**, tree + slot structure.

## Encoder design (numbers + text, per the request)

Per-type **entity encoders** (weights shared across entities of a type):
```
entity_embed = MLP( [ numeric stats (standardized)        # numbers
                    | tags multi-hot, partype/tree one-hot  # categories
                    | text_embedding(name + description) ] )# text (precomputed)
```
- **Text:** precompute sentence-embeddings of each entity's name+description **once per patch** with a small model (e.g. all-MiniLM-L6-v2), store as fixed vectors. Decouples text from training (cheap), can fine-tune later.
- **Patch-aware:** the numeric stats are the patch-specific ones → buffs/nerfs reflected; new champs/items encode from their stats (no cold-start ID problem).
- One encoder each for champion / item / rune / (later) monster.

## Integration into the models
- Replace/augment the current raw **ID embeddings** (`champion_id`, `keystone`, `primary_tree`) in 04f / the GNN with **patch-aware static embeddings** looked up by `(patch, id)`.
- **Prerequisite:** a `patch` feature per game — extract from raw `info.gameVersion` into features.parquet (easy; not present yet).
- **Items:** NOT in features currently (we have champ/runes/spells per slot, not item builds). Using the item encoder needs extracting per-player item builds from timeline `ITEM_PURCHASED/SOLD/UNDO` events — a new feature-processing step (heavier).
- **Monsters/objectives:** the timeline has `ELITE_MONSTER_KILL` / `BUILDING_KILL` events; a monster/objective encoder is most useful in a future *event-aware* model, less so for current snapshot models.

## Phased plan (cheapest, highest-value first)
1. **Patch feature** (extract `gameVersion` → `patch`) + **champion static encoder** (patch-aware champ embeddings). Champs already in features → test cross-patch generalization directly. ← start here
2. **Rune + summoner-spell static encoders** (also already in features).
3. **Item encoder** — requires timeline item-build extraction (new feature).
4. **Monster/minion/objective** — curated per-patch table + event-aware modeling.
- **Text embeddings:** precompute per patch (one-off) alongside step 1.

## Why it's worth it
Generalization across patches (critical once data spans many patches — exactly what
the all-elo + ongoing harvest produces), graceful handling of new champs/items
(cold-start), and richer semantics than opaque ID embeddings. Lowest-risk first win =
patch feature + champion static encoder; biggest data dependency = item builds.
