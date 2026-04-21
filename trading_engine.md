# Swing Trading Engine — System Overview

> A living design document for a semi-autonomous, multi-signal swing trading system.
> This is a spec for incremental development. The architecture is stable; the
> implementations behind each layer are expected to evolve continuously.

---

## 1. Vision & Mission

**Vision.** Build durable wealth through disciplined participation in the public
equity markets, and help others do the same by producing a system that
codifies what works.

**Mission.** Create a semi-autonomous swing trading system that researches,
screens, identifies entries and exits, and manages risk dynamically across
multiple signal types (technical, fundamental, macro, thematic, and catalyst).
The system makes recommendations; a human keeps final authority on new
entries. Over time the system deepens, but at every stage it must be usable,
auditable, and honest.

**Philosophy.** The edge is not in being right more often than wrong. It is in
asymmetric payoffs — small, fast losses and letting winners run. The system's
primary job is to enforce that discipline relentlessly.

---

## 2. Scope & Constraints

**In scope.**

- Long-only swing trading of U.S. equities and ETFs.
- Holding periods: days to months.
- Multi-signal: technical (Minervini / O'Neil style), fundamental (CANSLIM),
  macro, thematic, and catalyst-driven.
- Broker: Alpaca (paper account first, live later).
- Capital band: high five figures to low seven figures.

**Out of scope (non-negotiable).**

- No options.
- No shorting.
- No leverage beyond margin that an Alpaca cash/margin account permits for
  long equity.
- Not a fund. Not managing external capital. The system is a tool, not a
  service.

**Design constraints.**

- Incremental buildout — every phase must leave the system in a working,
  useful state.
- Evolvable — modules must be swappable without rewrites.
- Semi-autonomous — human approval required for new entries; exits can be
  fully automated once trusted.
- Paper-first — every new component runs against paper trading before
  going live.

---

## 3. Operating Principles

1. **Cut losses fast, let winners run.** The system must enforce initial
   stops rigidly and trail winners generously. Many small losses are
   acceptable. Giving back a big winner is not.

2. **Macro gates behavior, not ideas.** The market regime engine decides
   *whether* and *how aggressively* to trade. It does not filter which
   individual stocks are candidates.

3. **Macro tailwinds amplify conviction.** When a name passes the screens
   and its sector also has a supportive macro thesis, conviction and size
   increase. A macro tailwind is a multiplier, not a gate.

4. **Point-in-time correctness.** Every data point is tagged with the
   timestamp at which it became known. Backtests must only use
   information available at decision time.

5. **Stable contracts, disposable implementations.** Modules talk to each
   other through versioned schemas. Rewriting how a signal is produced
   must not break anything downstream.

6. **Layered human oversight.** The system always produces explanations,
   never just scores. Every recommendation carries a human-readable set of
   reasons and flags.

7. **Kill switches are primary features.** Daily loss caps,
   consecutive-loss circuit breakers, and a manual override are first-class
   components, not afterthoughts.

8. **One codebase for live and backtest.** The same signal functions run
   in production and in historical simulation. No separate backtest
   codepath.

---

## 4. System Architecture

### 4.1 High-Level Diagram

```
                        ┌───────────────────────────┐
                        │     Orchestration         │
                        │   (scheduler / cron)      │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │     Ingestion Layer       │
                        │  (pluggable providers)    │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │       Data Store          │
                        │  Parquet + DuckDB + RDB   │
                        │    (point-in-time)        │
                        └─────────────┬─────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
  ┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
  │  Per-Stock      │        │  Theme / Macro  │        │  Event /        │
  │  Funnel         │        │  Module         │        │  Catalyst       │
  │                 │        │                 │        │  Stream         │
  │  Universe → RS  │        │  Sector macro   │        │  News, filings, │
  │  → Fundamentals │        │  scores +       │        │  insider,       │
  │  → Technical    │        │  drivers        │        │  congressional, │
  │  → Event scrub  │        │                 │        │  unusual flow   │
  └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │      Watchlist Row        │
                        │    (the core contract)    │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │      Regime Engine        │
                        │    (global gate/dial)     │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │     Decision Engine       │
                        │  (portfolio + risk rules) │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │     Approval Queue        │
                        │   (human-in-the-loop)     │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │     Execution Layer       │
                        │     (Alpaca, bracket)     │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   State & Journal Store   │
                        └─────────────┬─────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │   Monitoring & Alerts     │
                        └───────────────────────────┘
```

### 4.2 Layers & Contracts

Each layer exposes a **stable schema** to the layer below it. Implementations
behind the schema can be replaced freely. The schemas are versioned; a bump
to a major version requires a migration, a minor bump is backward compatible.

---

## 5. Data Layer

**Purpose.** Single source of truth for every piece of information used by
any downstream module.

**Storage model.**

- **Raw zone** — raw API responses, immutable, append-only.
- **Curated zone** — cleaned, typed, schema-stable tables. This is what
  every consumer reads from.
- **Feature zone** — derived features computed from curated data (RS ranks,
  moving averages, ATRs, etc.), cached and refreshed on schedule.

**Data categories to ingest.**

- Price and volume (OHLCV) — daily first, intraday later.
- Fundamentals — income statement, balance sheet, cash flow, estimates,
  revisions.
- Earnings calendar.
- Corporate actions — splits, dividends, spinoffs.
- Macro — rates, yields, credit spreads, FX, commodities, VIX, economic
  releases (via FRED/ALFRED with revisions preserved).
- Market internals — advance/decline, new highs/lows, % above 50/200 MA.
- News — headline feed with ticker tagging.
- Alternative — insider filings, congressional trades, short interest,
  unusual options flow, ETF flows.

**Key discipline.** Every row carries:
- `as_of` — when the data point refers to.
- `observed_at` — when the system learned it.

This enables honest backtesting and correct replay.

---

## 6. Upstream Tracks (three in parallel)

Three independent tracks run in parallel and converge into the Watchlist Row.

### 6.1 Per-Stock Funnel

Narrows the tradeable universe to a small set of actionable names.

1. **Universe filter.** Liquidity (20-day avg dollar volume), market cap floor,
   exclude OTC / pink sheets / recent IPOs / leveraged ETFs.
2. **Leadership / relative strength.** Rank vs. SPY over 3/6/12 months;
   keep top quartile or top decile.
3. **Fundamental screen (CANSLIM layer).** EPS/sales acceleration, ROE,
   margin trends, institutional sponsorship, earnings surprise history.
4. **Technical setup (Minervini layer).** Trend template pass, Weinstein
   stage 2, base pattern (VCP, flat, cup/handle, high tight flag), base
   depth and length, volume behavior.
5. **Event scrub.** Flag or exclude names with imminent earnings,
   adverse news, or disqualifying catalysts.

### 6.2 Theme / Macro Module

Produces a macro score for each sector and industry. Not a filter; a
multiplier on conviction for names in that sector.

**Output shape:**

```
ThemeScore
├── sector                            # e.g., "Energy", "Semiconductors"
├── industry                          # finer grain
├── as_of
├── macro_score                       # 0.0 - 1.0
├── trend                             # rising / flat / falling
├── drivers: [
│     "WTI above 200MA, rising",
│     "OPEC cuts extended through Q3",
│     "Inventories 8% below 5yr avg"
│   ]
├── risks: [
│     "Demand data softening",
│     "Geopolitical headline risk"
│   ]
└── score_version
```

Inputs include sector-specific fundamentals, commodity prices, rate
sensitivity, policy context, and secular trend indicators (AI capex for
semis, rate cycle for financials, oil/gas balance for energy, etc.).

### 6.3 Event / Catalyst Stream

Continuous stream of per-name events that can either (a) generate a new
candidate for the funnel or (b) enrich an existing watchlist row.

Sources:
- News headlines (Benzinga, Tiingo, Alpaca news, etc.).
- SEC filings (8-K, 13F, Form 4 insider).
- Congressional trade disclosures (Quiver, Capitol Trades).
- Unusual options activity.
- Analyst revisions and rating changes.
- Social narrative momentum (fintwit, Reddit aggregates, used cautiously
  as a late signal, not a lead).

Each event is tagged with symbol, timestamp, type, magnitude, and a
normalized score.

---

## 7. Regime Engine

**Purpose.** Produce a single global score that answers: *given today's
tape, how aggressive should the system be overall?*

**Output shape:**

```
RegimeReading
├── as_of
├── regime_score                      # 0.0 - 1.0
├── regime_label                      # e.g., "confirmed_uptrend",
│                                     #       "uptrend_under_pressure",
│                                     #       "correction", "bear"
├── exposure_multiplier               # derived: 0.0 - 1.0
├── new_entries_allowed               # boolean
├── components: {
│     "breadth": 0.72,
│     "trend": 0.81,
│     "credit": 0.65,
│     "volatility": 0.58,
│     "policy": 0.55
│   }
├── drivers: [
│     "SPY > 200MA, 200MA rising",
│     "% stocks above 50MA: 64%",
│     "HY credit spread trending wider — caution"
│   ]
└── engine_version
```

**Component categories.**

- **Trend** — major index structure (SPY / QQQ vs. key MAs, follow-through
  and distribution day counts).
- **Breadth** — % stocks above 50/200 MA, new highs vs. new lows,
  advance/decline line, sector leadership dispersion.
- **Credit** — HY vs. IG spreads, term structure.
- **Volatility** — VIX level and VIX term structure.
- **Policy** — Fed meeting proximity, policy shock flags, geopolitical
  risk flags.

**How it is used.**

- `new_entries_allowed` is a hard boolean gate.
- `exposure_multiplier` scales the `suggested_position_size_pct` on every
  new trade.
- Regime shifts can also trigger tightening of stops on existing positions.

**How it is backtested.** Separately from strategies, through:
1. Classification quality against pre-labeled "bad tape" periods.
2. Conditional forward-return analysis bucketed by regime decile.
3. Strategy-conditioned backtests comparing CAGR, max drawdown, and
   Sortino with and without the overlay.
4. Walk-forward / out-of-sample testing.
5. Named-regime case studies (did it catch 2008, 2020, 2022, 2018 Q4?).

---

## 8. Watchlist Row — the core contract

The **Watchlist Row** is the central data object that flows from all three
upstream tracks into the Decision Engine. It is the interface contract for
the whole system. It is a **snapshot**, produced on a schedule (nightly by
default), and preserved historically for reproducibility.

**Design principles.**

- Denormalized: carries everything downstream needs, no joins required.
- Carries raw facts *and* derived scores.
- Carries pre-computed trade parameters (proposed stop, pivot, target).
- Carries human-readable reasons and flags.
- Versioned — every row is tagged with the rule versions that produced it.

**Schema.**

```
WatchlistRow
├── Identity
│   ├── symbol
│   ├── as_of
│   ├── watchlist_version
│   └── data_vintage
│
├── Liquidity & Tradeability
│   ├── avg_dollar_vol_20d
│   ├── market_cap
│   ├── float_shares
│   ├── short_interest_pct
│   └── days_to_cover
│
├── Leadership (RS)
│   ├── rs_rank                       # 0-99 IBD-style
│   ├── rs_line_new_high              # boolean
│   ├── return_1m / 3m / 6m / 12m
│   ├── sector_rs
│   └── industry_rs
│
├── Fundamentals (CANSLIM)
│   ├── eps_growth_last_q
│   ├── eps_growth_ttm
│   ├── eps_accelerating              # boolean
│   ├── sales_growth_last_q
│   ├── sales_growth_ttm
│   ├── roe
│   ├── margin_trend
│   ├── earnings_surprise_last_4q
│   ├── instit_sponsorship_trend
│   └── fundamental_score             # 0-100 composite
│
├── Technical Structure
│   ├── stage                          # Weinstein 1-4
│   ├── trend_template_pass            # boolean, all 8 checks
│   ├── price
│   ├── ma_50 / ma_150 / ma_200
│   ├── ma_stack_ok                    # 50 > 150 > 200
│   ├── ma_200_slope                   # rising / flat / falling
│   ├── dist_from_50ma_pct
│   ├── dist_from_52w_high_pct
│   └── atr_pct
│
├── Base / Pattern
│   ├── base_type                      # vcp / flat / cup_handle / htf / none
│   ├── base_depth_pct
│   ├── base_length_weeks
│   ├── pivot_price
│   ├── pivot_distance_pct
│   ├── contraction_count              # for VCP
│   └── volume_dry_up                  # boolean
│
├── Theme / Macro (joined from ThemeScore)
│   ├── sector_macro_score
│   ├── industry_macro_score
│   ├── macro_trend
│   └── macro_drivers                  # short list
│
├── Event / Catalyst
│   ├── next_earnings_date
│   ├── days_to_earnings
│   ├── recent_news_count_7d
│   ├── news_sentiment_7d              # -1..+1
│   ├── analyst_upgrades_30d
│   ├── insider_buy_cluster            # boolean
│   ├── congressional_trade_flag       # boolean
│   └── catalyst_score                 # 0-100
│
├── Trade Parameters (pre-computed)
│   ├── proposed_stop
│   ├── initial_risk_pct               # stop distance from pivot
│   ├── proposed_target_1              # +20-25% above pivot
│   ├── risk_reward
│   └── suggested_position_size_pct    # before regime multiplier
│
├── Composite Scoring
│   ├── setup_score                    # technical, 0-100
│   ├── fundamental_score              # 0-100
│   ├── catalyst_score                 # 0-100
│   ├── composite_score                # weighted
│   └── conviction_tier                # A / B / C
│
└── Transparency
    ├── reasons: [                     # human-readable bullets
    │     "EPS growth accelerated: 38% -> 62% YoY",
    │     "VCP — 3 contractions, volume drying up",
    │     "RS rank 94, RS line at new high",
    │     "Pivot at $182.40, currently 0.8% below",
    │     "Sector macro tailwind: AI capex / memory cycle"
    │   ]
    ├── flags: [
    │     "Earnings in 6 days"
    │   ]
    └── rule_versions: {
          "screen": "v2.3",
          "scoring": "v1.8",
          "theme": "v1.1"
        }
```

---

## 9. Decision Engine

A **thin** module. It does not analyze; it orchestrates.

**Inputs.** The current watchlist, the current regime reading, current
portfolio state, and risk rules.

**Logic.**

1. Filter to `conviction_tier >= B`.
2. Remove rows with disqualifying `flags` (e.g., earnings within 5 days,
   active lawsuit flag).
3. If `regime.new_entries_allowed == false`, produce no new tickets.
4. Sort by `composite_score`.
5. For each candidate:
   - Apply `suggested_position_size_pct × regime.exposure_multiplier`.
   - Check sector concentration limit.
   - Check total portfolio heat (sum of initial risk across open trades).
   - Check correlation cap against existing positions.
6. Produce trade tickets with pre-filled stop, target, and size.

**Outputs.** Trade tickets queued into the Approval Queue.

---

## 10. Approval Queue

The human-in-the-loop layer.

- Every proposed trade renders with:
  - Symbol, proposed size, entry trigger, stop, target, R:R.
  - `reasons` and `flags` from the watchlist row.
  - Sector macro context.
  - Recent chart snapshot.
- The operator approves, rejects, or modifies size.
- Approved tickets move to the Execution Layer.
- Unapproved tickets expire within the session (typically end-of-day).

Over time, specific trade types can be configured for auto-approval once
their track record earns trust. Exits are typically fully automated from
day one.

---

## 11. Execution Layer

**Broker abstraction.** A pluggable interface; Alpaca is the first
implementation. Same interface should work for future brokers.

**Responsibilities.**

- Place bracket orders (entry + stop + optional take-profit).
- Handle partial fills, rejections, retries.
- Translate trade tickets into broker-specific order formats.
- Reconcile fills back to the state store.
- Respect market hours; queue or delay appropriately.

**Execution mode flag.** Every run carries a mode: `paper` or `live`.
Identical codepath, different credentials and account.

---

## 12. State & Journal

Single source of truth for portfolio state.

**Tables.**

- `orders` — every order placed, with status and broker IDs.
- `fills` — every execution.
- `positions` — current and historical positions, opened and closed.
- `journal_entries` — narrative notes per trade (entry thesis, emotional
  state, post-trade review).
- `signals_log` — every signal produced, whether or not acted on.
- `regime_log` — every regime reading with timestamp.
- `watchlist_snapshots` — full watchlist rows archived by date.
- `audit_log` — every state change with actor (system or human) and reason.

**Storage.** Postgres (or SQLite at the smallest scale). Journal entries
and long-form reasons stored as text. Snapshots of watchlists written to
Parquet for analytics queries.

---

## 13. Backtesting Framework

**Principle.** The backtester runs the *same* signal functions that run in
production. There is no parallel backtest codepath.

**Capabilities.**

- Replay any date range with point-in-time data.
- Regime-conditioned performance breakdowns.
- Per-module sensitivity analysis.
- Walk-forward with periodic refits.
- Golden backtest — a fixed reference test that must not regress across
  code changes.

**Key metrics.**

- CAGR, max drawdown, Sharpe, Sortino, Ulcer Index.
- Hit rate, avg winner, avg loser, win-to-loss ratio.
- Expectancy per trade, expectancy per dollar-day at risk.
- Drawdown distribution, time-to-recovery.
- Regime-bucketed performance.

---

## 14. Monitoring & Alerts

- Daily summary: positions, P&L, open risk, regime, watchlist size by
  conviction tier.
- Real-time alerts: stop hits, breakouts pending approval, regime changes,
  error conditions.
- Dashboard: one Streamlit page backed by the state store for the first
  phase; Grafana for real-time views later.
- Alerts channel: email or Slack/Discord/Telegram webhook.

---

## 15. Governance & Safety

**Kill switches.**

- Daily realized loss cap — if hit, no new entries for the rest of the day.
- Weekly realized loss cap — if hit, no new entries for the rest of the
  week and regime thresholds tighten.
- Consecutive-loss circuit breaker — after N consecutive losses, force a
  review before new entries.
- Manual global kill — operator can freeze all new orders instantly.

**Audit.** Every trade, every signal, every regime reading is logged with
a version tag. Every configuration change is version-controlled.

**Data quality.** Every ingestion job emits quality flags (missing data,
stale data, outliers). A data-quality dashboard surfaces these.

---

## 16. Tech Stack (recommended starting picks)

Opinionated defaults. All of these are replaceable; the architecture does
not depend on any one choice.

- **Language.** Python 3.11+.
- **Config.** YAML files validated by Pydantic.
- **Secrets.** Local `.env` in early phases; secret manager when hosted.
- **Data storage.**
  - Parquet files partitioned by date for time-series and watchlist
    snapshots.
  - DuckDB as the analytical query engine over Parquet.
  - Postgres (or SQLite at the smallest scale) for stateful tables —
    orders, fills, positions, journal, audit log.
- **Orchestration.** Cron + scripts initially. Prefect or Dagster once
  there are multiple interdependent jobs. Dagster's asset model maps
  well to this domain.
- **Broker SDK.** `alpaca-py`.
- **Market & fundamentals data.** Alpaca (equities), FRED/ALFRED (macro),
  plus one of FMP / Finnhub / SimFin / Tiingo for fundamentals and news.
  Swappable behind a provider interface.
- **Testing.** pytest for unit tests; a "golden backtest" reference run
  as a regression test.
- **Dashboard.** Streamlit for the first internal UI.
- **Alerts.** Email (SMTP) or a Slack/Discord/Telegram webhook.
- **Version control.** Git, monorepo, feature branches.
- **Hosting.** Local machine first; small VPS (e.g., Hetzner) once
  continuous running is needed. Containerization deferred.

---

## 17. Incremental Build Path

Each phase must produce a *working* system — not a half-built one.

**Phase 0 — Plumbing.**
- Repo, Python environment, config loader, logger.
- Alpaca paper account wired up.
- Data store with one source (Alpaca daily bars).
- Smoke test: place and cancel a paper order through the code.

**Phase 1 — One signal, end-to-end.**
- Minervini trend template screen on daily bars.
- Pivot breakout entry.
- Hardcoded position size (e.g., 1% risk per trade).
- Hardcoded initial stop (below pivot / 7-8%).
- Trade tickets queued to email for manual approval.
- Runs on paper for at least several weeks.

**Phase 2 — Risk and exits.**
- Exit manager: trailing stop, MA break, partial profit take.
- Portfolio heat tracker.
- Daily / weekly loss caps.
- Structured journal entries.

**Phase 3 — Fundamentals.**
- Fundamentals provider integration.
- CANSLIM-style screen.
- Watchlist row becomes a first-class object (schema frozen here).

**Phase 4 — Regime engine.**
- Simple version: 4-5 indicators, equal weight, single 0-1 score.
- Position-size multiplier wired into the decision engine.
- Regime log persisted.

**Phase 5 — Backtesting framework.**
- Use existing signal functions unchanged.
- Walk-forward and golden backtest established.
- Regime-conditioned performance views.

**Phase 6 — Theme / macro module.**
- Per-sector macro scores.
- Joined into watchlist rows.
- Conviction and size adjustments.

**Phase 7 — Event / catalyst stream.**
- News, filings, insider, congressional.
- Idea generation pipe into the funnel.
- Per-name de-risking flags.

**Phase 8 — Operational hardening.**
- Hosted on VPS, reliable scheduling, observability.
- Dashboard matured.
- Selective auto-approval for trusted trade types.

**Phase 9+ — Continuous evolution.**
- Alternative data, deeper regime modeling, ML overlays, richer UI,
  multi-account support, strategy variants.

---

## 18. What This System Is Not

- Not a fund. Not for external capital.
- Not a day trading system.
- Not an options or derivatives platform.
- Not a short-selling platform.
- Not a fully autonomous agent — the human always has veto on new
  entries.
- Not a machine-learning-first system. ML may be layered in later, but
  the core signals are interpretable rules.
- Not a Python research notebook. Notebooks are used for research;
  production runs from versioned code.

---

## 19. Open Questions / To Fill In Later

- Final weighting for `composite_score` across fundamental / setup /
  catalyst / macro. To be set empirically from backtest.
- Exact threshold for `conviction_tier` A vs. B vs. C.
- Correlation-cap policy (which correlation window, threshold).
- Auto-approval eligibility rules per trade type.
- Tax-lot management policy (FIFO, LIFO, specific lots).
- Rebalancing and cash-management policy during extended uptrends.
- Intraday data strategy (when and for what purpose).
- Treatment of earnings holds vs. earnings-proximity exits.

---

## 20. Glossary

- **CANSLIM** — William O'Neil's growth-stock framework (Current earnings,
  Annual earnings, New products/management, Supply/demand, Leader,
  Institutional sponsorship, Market direction).
- **Minervini Trend Template** — Mark Minervini's eight technical
  checkpoints a stock must satisfy to qualify as a Stage 2 uptrend
  candidate.
- **VCP** — Volatility Contraction Pattern.
- **Stage analysis** — Stan Weinstein's four-stage market cycle (base /
  advance / top / decline).
- **Follow-through day** — O'Neil's market confirmation signal for a new
  uptrend.
- **Distribution day** — a heavy-volume down day on the major indices,
  indicative of institutional selling.
- **RS rank** — Relative Strength percentile, typically 0-99.
- **R** — risk unit; one R is the distance from entry to initial stop,
  used for sizing and performance measurement.

---

*This document defines the architecture and scope. It does not prescribe
a single implementation. Modules may be rewritten freely as long as their
schemas remain backward compatible.*

