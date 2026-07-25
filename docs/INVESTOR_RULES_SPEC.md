# INVESTOR RULES SPEC — Codifiable Methods of History's Greatest Investors

Deterministic rule extraction for the multi-asset macro trading engine.
Every rule below is expressed as: **testable condition + exact threshold + required data**.
Vague philosophy is excluded by design. Where a master never published an exact number,
the canonical codification used by practitioners is given and flagged `[codified]`.

Confidence tags: **[HIGH]** = published by the investor or in their canonical book.
**[MOD]** = widely accepted codification of their stated practice. **[LOW]** = inferred, use as default parameter only.

---

## 1. Warren Buffett / Charlie Munger — Quality-Moat Value

### Sources
Berkshire shareholder letters (esp. 1986 "owner earnings" appendix), *Buffettology* (M. Buffett),
*Warren Buffett and the Interpretation of Financial Statements*, *Poor Charlie's Almanack*.

### CODIFIABLE RULES

| # | Rule | Exact Threshold | Data Needed | Conf |
|---|------|-----------------|-------------|------|
| B1 | Consistent high ROE (moat proxy) | ROE ≥ 15% in **each** of the last 10 fiscal years (no single year below 15%) | 10y income statements + equity | HIGH |
| B2 | High return on invested capital | ROIC ≥ 12% avg over 10y | NOPAT, invested capital | MOD |
| B3 | Low leverage | LT debt / net income ≤ 4.0 (debt repayable from ≤4 years of earnings); alt: debt/equity ≤ 0.5 | Balance sheet, net income | MOD |
| B4 | Durable gross margin | Gross margin ≥ 40% every year for 10y | 10y income statements | HIGH (book codification) |
| B5 | SG&A discipline | SG&A ≤ 30% of gross profit | Income statement | HIGH (book codification) |
| B6 | Low capital intensity | Depreciation ≤ 10% of gross profit AND capex ≤ 50% of net income (10y avg) | Cash flow statement | HIGH (book codification) |
| B7 | Interest coverage | Interest expense ≤ 15% of operating income | Income statement | HIGH (book codification) |
| B8 | Net margin | Net margin ≥ 20% sustained | Income statement | MOD |
| B9 | Owner earnings (the valuation input) | OE = NetIncome + D&A + non-cash charges − maintenance_capex − ΔWC. Proxy maintenance_capex = avg(capex, 5y) × (1 − revenue_growth_capex_share); simplest deterministic proxy: maintenance_capex = D&A | Full financials, 5–10y | HIGH (1986 letter) |
| B10 | Intrinsic value | Two-stage DCF of owner earnings: stage-1 growth g1 = min(10y OE CAGR, 15%) for 10 years, terminal g2 = 2.5%, discount rate r = max(10-yr Treasury + 3%, 9%) | OE series, treasury yield | MOD (parameters codified) |
| B11 | Margin of safety | BUY only if Price ≤ 0.70 × IV (30% MOS); STRONG BUY ≤ 0.50 × IV. SELL/refuse if Price ≥ 1.10 × IV | IV from B10, market price | MOD (Graham's one-third, Buffett unquantified) |
| B12 | Earnings predictability gate (circle of competence) | Linear regression of ln(EPS) on 10y: R² ≥ 0.80 AND no more than 1 down-year > 10% decline. Fail → stock is untouchable regardless of cheapness | 10y EPS | MOD |
| B13 | Retained-earnings test | Rolling 5y: ΔMarketCap ≥ Σ(retained earnings over same 5y). Fail → management destroys value → exclude | 5y market cap, retained earnings | HIGH (letters) |
| B14 | Buyback quality | Buybacks count as positive signal only when executed at Price < IV; buybacks above IV score negative | Buyback history, IV | HIGH |
| B15 | Never sell on price alone | Exit triggers are: rule B1/B4/B13 breaks on new annuals, or price ≥ 1.5 × IV. No stop-loss on this sleeve | Ongoing fundamentals | MOD |
| B16 | Concentration | Max 5–10 positions in the quality sleeve; min position 10% of sleeve if all filters pass ("punch card") | Portfolio state | MOD |

### Munger — Inversion + Psychological Misjudgment (mechanized)

Munger's list is behavioral, but the engine can enforce the *anti-bias* mechanically:

| # | Bias (Munger) | Mechanical Enforcement | Threshold / Check |
|---|---------------|------------------------|-------------------|
| M1 | Incentive-caused bias | Screen management incentives | Insider ownership ≥ 5% OR exec comp ≥ 50% tied to ROIC/EPS; REJECT if stock-based comp > 5% of revenue |
| M2 | Commitment & consistency | Forced re-underwrite | Every position re-scored from scratch every 90 days with entry thesis hidden from the scorer function (recompute all filters blind) |
| M3 | Deprival / loss aversion | Cost-basis blindness | Position-management functions receive current price + thesis only; **entry price must not be an input** to hold/exit logic (except explicit stop rules) |
| M4 | Social proof | Crowding filter | REJECT new entry if short interest > 20% of float being squeezed, or if position appears in > 5% of 13F top-10s [equities]; for crypto: funding rate > 0.1%/8h = crowded long |
| M5 | Availability / recency | Base-rate anchor | Any forecast input (growth, vol) is shrunk 50% toward the 20y asset-class base rate before use |
| M6 | Inversion ("invert, always invert") | Pre-mortem kill-list | Entry requires computing the *failure* checklist: list of N conditions that would make trade wrong; each must have a measurable trigger stored with the trade (used later by attribution, §9) |
| M7 | Lollapalooza (confluence) | Signal stacking | Conviction score = count of independent passing filters; size scales only when ≥ 3 independent factor families agree (fundamental + technical + macro) |

---

## 2. Benjamin Graham — Deep Value / Statistical Cheapness

### Sources
*Security Analysis*, *The Intelligent Investor* (ch. 14), Graham-Rea 10-point list (1977).

### CODIFIABLE RULES

**A. Net-Net (NCAV) strategy — Graham's most mechanical system**

| # | Rule | Exact Threshold | Data Needed | Conf |
|---|------|-----------------|-------------|------|
| G1 | NCAV definition | NCAV = Current Assets − Total Liabilities − Preferred Stock | Balance sheet | HIGH |
| G2 | Buy price | Price ≤ 0.667 × NCAV per share | NCAV, price | HIGH |
| G3 | Quality overlay | Positive trailing 12m earnings AND currently paying any dividend (optional relaxation: earnings only) | Income statement | HIGH |
| G4 | Diversification mandate | ≥ 30 simultaneous net-net positions, equal-weighted; single name ≤ 3.3% of sleeve | Portfolio | HIGH |
| G5 | Exit | Sell at Price ≥ NCAV (≈ +50% from entry) OR after 2 years holding, whichever first | Price, holding time | HIGH (Graham's stated practice) |

**B. Graham Number (fair value ceiling for defensive stocks)**

| # | Rule | Exact Threshold | Data Needed |
|---|------|-----------------|-------------|
| G6 | Graham Number | GN = √(22.5 × EPS_ttm × BVPS). Buy zone: Price ≤ GN. (22.5 = 15 P/E × 1.5 P/B) | EPS, book value/share |

**C. Defensive Investor — the 7 criteria (Intelligent Investor ch. 14)**

| # | Criterion | Exact Threshold (1973 → inflation-adjusted 2026) | Data Needed |
|---|-----------|--------------------------------------------------|-------------|
| G7 | Adequate size | Sales ≥ $100M industrial / $50M utility (1973$) → codify as **revenue ≥ $700M** or **market cap ≥ $2B** | Revenue / mkt cap |
| G8 | Financial condition | Current ratio ≥ 2.0 **AND** LT debt ≤ net current assets (working capital) | Balance sheet |
| G9 | Earnings stability | Positive EPS in **every** one of the last 10 years | 10y EPS |
| G10 | Dividend record | Uninterrupted dividend payments for 20 consecutive years | Dividend history |
| G11 | Earnings growth | 10y EPS growth ≥ +33% total, measured as (avg EPS yrs 8–10) / (avg EPS yrs 1–3) ≥ 1.33 (≈ 2.9%/yr) | 10y EPS |
| G12 | Moderate P/E | Price ≤ 15 × average EPS of trailing 3 years | 3y EPS, price |
| G13 | Moderate P/B | Price/Book ≤ 1.5, with escape valve: **P/E × P/B ≤ 22.5** allowed | Book value |

**D. Graham-Rea 10-point bonus screen (each point = 1; buy if reward-points ≥ 1 of {1,3,5} AND safety-points ≥ 1 of {6,7,8})**

| # | Point | Threshold |
|---|-------|-----------|
| GR1 | Earnings yield ≥ 2 × AAA corporate bond yield |
| GR2 | P/E ≤ 40% of stock's own 5y-high P/E |
| GR3 | Dividend yield ≥ 2/3 × AAA bond yield |
| GR4 | Price ≤ 2/3 × tangible book value per share |
| GR5 | Price ≤ 2/3 × NCAV per share |
| GR6 | Total debt < book value |
| GR7 | Current ratio ≥ 2 |
| GR8 | Total debt ≤ 2 × NCAV |
| GR9 | 10y EPS CAGR ≥ 7% |
| GR10 | ≤ 2 year-over-year EPS declines of ≥ 5% in past 10 years |

---

## 3. Jesse Livermore — Pivotal Points, Confirmation, Pyramiding

### Sources
*How to Trade in Stocks* (1940, incl. the Livermore Market Key), *Reminiscences of a Stock Operator*.

### CODIFIABLE RULES

| # | Rule | Exact Threshold | Data Needed | Conf |
|---|------|-----------------|-------------|------|
| L1 | Trade only the line of least resistance | Only long when instrument makes higher swing-highs & higher swing-lows over lookback (trend filter); only short the inverse. Never counter-trend | OHLC daily | HIGH |
| L2 | Pivotal point = prior significant extreme | Pivot(long) = max(high, N=52-week or consolidation range high). Entry: close > pivot | OHLC | HIGH |
| L3 | Continuation pivotal point | After a trend leg, a consolidation of ≥ 3 weeks with range ≤ 15%; breakout of that range = re-entry/add point | OHLC | MOD |
| L4 | Market Key swing filter | For stocks > $30 (1940): record only moves ≥ 6 points ≈ **modernize as swing filter of 2.5 × ATR(20) or 6% of price, whichever larger**; a "natural reaction" = counter-move of that magnitude | OHLC, ATR | HIGH (numbers), MOD (modernization) |
| L5 | Confirmation / penetration rule | Trend reversal or breakout is confirmed only when price penetrates the pivotal point by ≥ 3 points on $30+ stock → **modernize: ≥ 0.5 × ATR(20) or 1% beyond pivot** | OHLC | HIGH / MOD |
| L6 | Abnormal reaction = exit | If after breakout price falls back below pivot within the confirmation window, the move is FALSE → exit immediately, full size | Intraday/daily close vs pivot | HIGH |
| L7 | Time stop ("acting right") | If position not profitable within 3 sessions of entry, cut it | Trade log | MOD |
| L8 | Probe-and-pyramid sizing | Split intended full size into tranches: **20% / 20% / 20% / 40%**. Tranche 1 at pivot break; each subsequent tranche only if price ≥ prior fill + 0.5 × ATR (i.e., only add at HIGHER prices for longs). If tranche 1 loses, never add | Fills, price | HIGH (20% probe stated in book; 20/20/20/40 split is his cotton example) |
| L9 | Never average down | Adding to a losing position is prohibited at engine level (hard risk-engine veto) | Position P&L | HIGH |
| L10 | 10% bucket-shop stop | Max loss on any single trade = 10% of position value — absolute ceiling; tighter stops from other modules take precedence | Entry, price | HIGH |
| L11 | 50% retracement signal | After a major move, a retracement of 50% of the move is the key reference: reversal from the 50% level in trend direction = re-entry signal; decisive break beyond it = trend over | Swing highs/lows | HIGH |
| L12 | Profit reserve | After closing a winning campaign, sweep 50% of realized profit to non-tradable cash reserve | Realized P&L ledger | HIGH (his stated rule, honored in the breach) |
| L13 | Sit tight — exit on trend break only | Winners are exited only by L4-scale reversal signal or trailing swing-low break, never on fixed profit target | Swing structure | HIGH |

---

## 4. Stanley Druckenmiller / George Soros — Reflexivity, Liquidity, Concentration

### Sources
Soros *The Alchemy of Finance*; Druckenmiller interviews/speeches (Lost Tree Club 2015, *New Market Wizards*).

### CODIFIABLE RULES

| # | Rule | Exact Threshold | Data Needed | Conf |
|---|------|-----------------|-------------|------|
| D1 | Liquidity is the master signal | Net Fed liquidity = Fed balance sheet − TGA − RRP. Risk-on gate: 13-week change > 0. Crypto version: stablecoin aggregate mcap 30d change > 0 AND global M2 YoY rising | FRED (WALCL, WTREGEN, RRPONTSYD), stablecoin supply | MOD (his stated framework, thresholds codified) |
| D2 | Reflexivity boom detector (self-reinforcing phase) | Price 12m ROC > +20% **AND** the fundamental it feeds is accelerating (credit growth 2nd derivative > 0, or for crypto: active addresses/TVL 90d slope > 0). Both up = ride the bubble, do NOT short | Price, credit/on-chain fundamentals | MOD |
| D3 | Reflexivity bust detector (twilight phase) | Price makes new 6m high but fundamental series fails to confirm (negative divergence over 2 consecutive quarters) AND price then breaks 20-week MA → short/exit signal. Never short on valuation alone | Same as D2 | MOD |
| D4 | Asymmetry filter (Soros) | Enter only if modeled reward:risk ≥ 3:1 (distance to target vs distance to invalidation stop) | Target, stop levels | MOD |
| D5 | Probe first, investigate later | Initial position = 25% of intended size ("test the thesis with money"); scale to full only after price confirms (moves ≥ 1 ATR in favor) | Fills | HIGH (practice), MOD (numbers) |
| D6 | Bet big on confluence | When fundamental + technical + liquidity (D1) all align: position may reach 20–30% of book risk-weighted (Druckenmiller: "only 1–2 great ideas a year") | Signal scores | HIGH (practice), MOD (cap) |
| D7 | Never trade against a strong trend | Block counter-trend entries when |price − 200d MA| / 200d MA > 20% and ADX(14) > 30 in trend direction | OHLC | MOD |
| D8 | Drawdown circuit breaker | Monthly P&L −5% → cut gross exposure 50%. −10% → flat, mandatory 2-week cooloff, restart at 25% gross. Rationale: Druckenmiller's "when you're down, get small" | Equity curve | MOD (his stated behavior, thresholds codified) |
| D9 | House-money pressing | Allowed gross exposure scales with YTD P&L: base 100%; if YTD > +10%, max gross 150%; if YTD < 0, max gross 70% | Equity curve | MOD |
| D10 | Thesis invalidation stop | Every entry stores an invalidation price/condition (from M6 pre-mortem). Hit = exit, no debate, no re-entry same direction for 5 sessions | Trade metadata | HIGH (discipline), MOD (5d) |
| D11 | Central-bank pivot trade | Direction change in policy-rate expectations (2y yield 20d change crossing zero) = highest-weight macro input for FX/rates/crypto sleeves | 2y yields, futures-implied rates | MOD |

---

## 5. Jim Simons / Renaissance — Signal Hygiene (public knowledge only)

### Sources
Zuckerman *The Man Who Solved the Market*; public interviews of Simons, Brown, Mercer, Patterson. NOTE: actual Medallion signals are secret; only *process* rules are codifiable, which is exactly what an engine needs.

### CODIFIABLE RULES

| # | Rule | Exact Threshold | Data Needed | Conf |
|---|------|-----------------|-------------|------|
| S1 | Many small edges, no hero signals | Portfolio must hold ≥ 20 concurrent independent signals; no single signal > 10% of total risk budget; target per-signal hit rate 50.5–52% is acceptable if N is large | Signal registry | HIGH (publicly stated: "right 50.75% of the time") |
| S2 | Short holding periods | Signal library skews short: median intended holding period ≤ 5 days for the stat sleeve; capacity and decay checked per horizon | Trade log | HIGH |
| S3 | Admission test for any new signal | Backtest t-stat ≥ 3.0 (or Probabilistic Sharpe Ratio ≥ 0.95 vs 0 benchmark) **AND** out-of-sample walk-forward Sharpe ≥ 0.5 × in-sample Sharpe **AND** survives transaction-cost model at 2× estimated costs | Backtest engine | MOD (canonical quant practice attributed to them) |
| S4 | Never override the model | Manual intervention only via kill-switch (flatten), never discretionary position-taking. Engine enforces: no order path outside signal pipeline | Architecture | HIGH (famously enforced) |
| S5 | Kill decaying signals | Retire signal when rolling 90-day Sharpe < 0 AND rolling 252-day Sharpe < 0.3, OR CUSUM drift test on daily signal P&L rejects at 95% confidence. Retired → quarantine 90 days → must re-pass S3 on fresh data | Per-signal P&L series | MOD |
| S6 | Correlation caps | Pairwise 90d correlation of any two live signals' P&L ≤ 0.7; if breached, keep the higher-Sharpe one at full weight, halve the other | Signal P&L matrix | MOD |
| S7 | Cost-aware execution | Expected edge per trade must exceed 2 × (spread/2 + fee + modeled impact). Impact model: k × (order_size/ADV)^0.5 | Microstructure data | MOD |
| S8 | Anomaly ≠ explanation required, but persistence is | A signal without economic rationale gets half the risk budget of one with rationale; both must show persistence across ≥ 2 non-overlapping historical sub-periods and ≥ 2 instruments | Backtest metadata | MOD (they traded non-intuitive signals but weighted carefully) |
| S9 | Book-level vol targeting | Scale gross daily so realized 20d portfolio vol → target (e.g., 15% annualized); recompute daily | Returns | MOD |
| S10 | Data integrity first | Any bar failing sanity checks (price jump > 20σ, negative volume, stale timestamp) is quarantined; signals never fire on unvalidated data | Raw feeds | HIGH (their stated obsession with clean data) |

---

## 6. Ray Dalio — Regime Detection + Risk Parity

### Sources
Bridgewater "All Weather Story", *Principles*, *Big Debt Crises*, public Bridgewater research.

### CODIFIABLE RULES

**A. Growth/Inflation quadrant regime detector**

| # | Rule | Exact Threshold | Data Needed |
|---|------|-----------------|-------------|
| R1 | Growth axis | GROWTH_UP if ≥ 2 of 3: (a) ISM PMI > 50 and 3m slope > 0; (b) 4-wk avg initial claims 13-wk change < 0; (c) GDP nowcast > consensus. Else GROWTH_DOWN | ISM, claims, nowcast (FRED/Atlanta Fed) |
| R2 | Inflation axis | INFL_UP if ≥ 2 of 3: (a) CPI YoY 3m momentum > 0; (b) 10y breakeven 3m change > +10bp; (c) commodity index (BCOM) > 200d MA. Else INFL_DOWN | CPI, breakevens, BCOM |
| R3 | Quadrant → asset tilt | Q1 G↑I↓: equities, credit, crypto-beta. Q2 G↑I↑: commodities, EM, BTC-as-commodity, TIPS. Q3 G↓I↓: nominal long bonds, USD, quality. Q4 G↓I↑: gold, TIPS, cash, trend-following. Each quadrant gets 25% of *risk*, tilted +50%/−50% by detected quadrant | R1, R2 outputs |
| R4 | Regime persistence filter | Quadrant must persist 2 consecutive monthly reads before strategy weights shift (anti-whipsaw) | Regime history |
| R5 | Debt-cycle overlay | Deleveraging warning when: private credit/GDP 5y change > +15pp AND debt-service ratio rising; if triggered, cap gross at 75% and raise Q3/Q4 weights | BIS credit stats |

**B. Risk parity sizing**

| # | Rule | Exact Threshold | Data Needed |
|---|------|-----------------|-------------|
| R6 | Naive risk parity | weight_i = (1/σ_i) / Σ(1/σ_j), σ = 60d realized vol, EWMA λ=0.94 | Returns |
| R7 | Equal risk contribution (full) | Solve w such that w_i(Σw)_i = (1/N) wᵀΣw for all i (Newton/cyclical coordinate descent); rebalance monthly or when any RC drifts > 20% from parity | Covariance matrix |
| R8 | Vol-target leverage | Lever the parity portfolio to 10–12% annualized target vol; leverage cap 2.0× (crypto sleeve cap 1.0×) | Portfolio vol |
| R9 | All Weather static fallback | If data pipeline degraded: 30% equities / 40% long bonds / 15% intermediate bonds / 7.5% gold / 7.5% commodities (Dalio's published retail version) | None (static) |
| R10 | Correlation-shock brake | If avg pairwise 20d correlation across sleeves > 0.8 (everything-selling-off), cut leverage to 0.5× until it drops below 0.6 | Correlation matrix |

---

## 7. Mark Minervini / William O'Neil — SEPA & CANSLIM

### Sources
Minervini *Trade Like a Stock Market Wizard*, *Think & Trade Like a Champion*; O'Neil *How to Make Money in Stocks* (4th ed.). These two are the most numerically explicit sources in this document.

### CODIFIABLE RULES

**A. Minervini Trend Template — ALL 8 must be true to qualify [HIGH — verbatim from his book]**

| # | Criterion | Exact Threshold |
|---|-----------|-----------------|
| T1 | Price > 150d MA AND Price > 200d MA |
| T2 | 150d MA > 200d MA |
| T3 | 200d MA rising for ≥ 1 month (prefer ≥ 4–5 months) |
| T4 | 50d MA > 150d MA AND 50d MA > 200d MA |
| T5 | Price > 50d MA |
| T6 | Price ≥ 1.30 × 52-week low (at least 30% above) |
| T7 | Price ≥ 0.75 × 52-week high (within 25% of high; closer is better) |
| T8 | Relative Strength rank ≥ 70 (percentile vs universe over 12m, weighted 40% on last quarter); prefer ≥ 80–90 |

**B. Minervini VCP (Volatility Contraction Pattern) [HIGH]**

| # | Rule | Exact Threshold |
|---|------|-----------------|
| V1 | Base duration 3–65 weeks |
| V2 | 2–6 progressive contractions ("T's"); each contraction depth ≈ ≤ 0.5 × previous (e.g., 25% → 12% → 6% → 3%) |
| V3 | Final contraction depth ≤ 10% (ideally ≤ 5%) |
| V4 | Volume in final contraction < 50-day average volume ("volume dry-up", ≥ 1 day below 50% of avg) |
| V5 | Buy = breakout above pattern pivot (high of final contraction); breakout-day volume ≥ 1.5 × 50d avg |
| V6 | Prior uptrend ≥ 30% before base forms (leadership prerequisite) |

**C. Minervini risk management [HIGH — his stated numbers]**

| # | Rule | Exact Threshold |
|---|------|-----------------|
| V7 | Initial stop: never > 10% below entry; normal 5–8%; average loss should be ≤ 6% |
| V8 | Risk per trade ≤ 1.25% of equity (position_size = 0.0125 × equity / stop_distance%) |
| V9 | Max single position 20–25% of book; typical 10–15%; min meaningful 5% |
| V10 | Reward:risk at entry ≥ 2:1 (target vs stop) |
| V11 | Move stop to breakeven when gain ≥ 2 × initial risk; sell 1/3–1/2 into strength at gain = 2–3 × risk |
| V12 | Win-rate adaptive throttle: if last 10 trades win-rate < 40%, halve all new position sizes; < 30%, stop trading, paper-trade until 2 consecutive winners |
| V13 | Progressive exposure: after drawdown, re-enter at 25–50% normal size; scale to 100% only after net profitable trades |

**D. O'Neil CANSLIM screen [HIGH — from the book, exact numbers]**

| Letter | Rule | Exact Threshold | Data Needed |
|--------|------|-----------------|-------------|
| C | Current quarterly EPS growth | EPS YoY ≥ +25% (prefer +40–100%); accelerating vs prior quarters; quarterly sales ≥ +25% YoY or accelerating 3 quarters | Quarterly statements |
| A | Annual EPS growth | ≥ +25%/yr in each of last 3 years; ROE ≥ 17%; annual EPS in each year higher than prior | Annual statements |
| N | New highs / new catalyst | Price within 15% of 52-wk high or breaking to new high from a valid base; identifiable new product/mgmt (binary catalyst flag) | Price, news flag |
| S | Supply & demand | Volume on up-days > volume on down-days (50d up/down volume ratio > 1.0); net buybacks positive; smaller float scores higher | Volume, float, buybacks |
| L | Leader | RS rating ≥ 80 (prefer ≥ 90); REJECT if RS < 70; stock among top 2–3 in industry group by RS; industry group in top 25% of groups | Universe returns |
| I | Institutional sponsorship | Number of funds owning increased in latest 2 quarters; ≥ 1 top-decile-performing fund holds it; but REJECT if > 60% institutional ownership (over-owned) | 13F data |
| M | Market direction (gates everything) | Buy only in "confirmed uptrend": **follow-through day** = day 4–10 of rally attempt, major index +1.25% or more on volume > prior day. Market top: **5 distribution days within 25 sessions** (down ≥ 0.2% on higher volume) → move to cash/defense | Index OHLCV |

**E. O'Neil cup-with-handle specification [HIGH]**

| # | Element | Exact Spec |
|---|---------|-----------|
| CH1 | Prior uptrend ≥ 30% before base |
| CH2 | Base length 7–65 weeks (minimum 7) |
| CH3 | Cup depth 12–33% from high (max 50% only in bear-market bases) |
| CH4 | Cup shape: rounded "U" — time at lows ≥ 2 weeks (reject sharp "V": low-to-rim recovery < 2 weeks) |
| CH5 | Handle: forms in upper half of base AND above the 10-week (50d) MA |
| CH6 | Handle depth ≤ 12% (typically 8–12%); drift downward along lows; duration ≥ 5 sessions |
| CH7 | Handle volume: declining, at least one day < 50% of 50d avg volume |
| CH8 | Pivot buy point = handle high + 0.1% (O'Neil: 10 cents); valid buy zone = pivot to pivot × 1.05 (never chase > 5% past pivot) |
| CH9 | Breakout volume ≥ 1.4–1.5 × 50d average volume; weak volume breakout (< 1.0×) = suspect, sell into failure |
| CH10 | **Hard stop 7–8% below purchase price — no exceptions ever** (O'Neil's cardinal rule) |
| CH11 | Take profits at +20–25% for most trades, EXCEPT: if stock gains ≥ 20% within 3 weeks of breakout → hold minimum 8 weeks (potential big winner) |

---

## 8. Ed Thorp — Kelly Sizing in Practice

### Sources
*Beat the Dealer*, *Beat the Market*, "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market" (Thorp 2006), *A Man for All Markets*.

### CODIFIABLE RULES

| # | Rule | Exact Formula / Threshold | Data Needed | Conf |
|---|------|---------------------------|-------------|------|
| K1 | Discrete Kelly | f* = (b·p − q)/b where p=win prob, q=1−p, b=win/loss payoff ratio | Trade-level win rate & payoff from module's rolling stats (min 50 trades) | HIGH |
| K2 | Continuous Kelly (assets) | f* = (μ − r)/σ² (fraction of capital, can exceed 1 = leverage) | Expected excess return, variance | HIGH |
| K3 | Multi-asset Kelly | f = Σ⁻¹(μ − r1) (vector), with Σ shrunk via Ledoit-Wolf | Covariance, expected returns | HIGH |
| K4 | **Fractional Kelly — the practical rule** | Bet 0.25–0.50 × f*. Engine default: **0.25 Kelly** for signals < 1y live, **0.50 Kelly** for seasoned signals. Never > 0.5. Half-Kelly ≈ 75% of growth rate at ~50% of the variance | f* from K1–K3 | HIGH (Thorp explicitly used ≤ ½ Kelly at Princeton Newport) |
| K5 | Edge shrinkage before Kelly | Shrink estimated μ (or p) 50% toward zero (or toward 50%) before computing f* — estimation error is the Kelly killer; overbetting true Kelly is catastrophic, underbetting merely suboptimal | Signal stats | HIGH (his stated asymmetry argument) |
| K6 | Drawdown math as constraint | Full Kelly ⇒ P(ever halving bankroll) = 50%; P(dropping to 1/n) = 1/n. Choose fraction c so P(drawdown > D_max) ≤ tolerance: at fraction c, P(ever dropping to fraction x of peak) = x^(2/c − 1). Solve for c given board-level D_max (e.g., 20% ⇒ c ≈ 0.36) | Risk policy | HIGH (Thorp 2006 formulas) |
| K7 | Per-position cap regardless of Kelly | No position > 2% risk of equity for stat-arb-style sleeves (his PNP practice: ~2% positions, hundreds of names); concentrated macro sleeve may go higher only via D6 confluence rule | Portfolio | HIGH |
| K8 | Correlated-bet reduction | If simultaneous positions have pairwise ρ > 0.5, treat as one bet: sum their Kelly fractions and cap at single-bet f* | Correlations | MOD |
| K9 | Recompute cadence | Re-estimate p, b, μ, σ on rolling 100-trade / 252-day window; Kelly fraction updates only at rebalance, not intra-trade | Trade log | MOD |

---

## 9. THE SELF-LEARNING LOOP — Attribution, Decay, Regime Weighting

How the above rules become a *Karpathy-loop* self-improving system. Three mechanisms:

### 9.1 Post-trade attribution — "which factor was wrong"

Every trade MUST be stored at entry with its full causal record:

```
TradeRecord {
  signal_id, timestamp, direction, size,
  factor_vector: {trend_template_score, vcp_quality, canslim_score,
                  liquidity_regime (D1), quadrant (R1-R3), rs_rank,
                  kelly_inputs (p, b, μ, σ), conviction_count (M7)},
  premortem: [ {condition, trigger_level} ],   # from M6
  expected: {edge_bps, reward_risk, hold_days, invalidation_price}
}
```

At close, compute **attribution decomposition** (all deterministic):

| Component | Formula | What it isolates |
|-----------|---------|------------------|
| Thesis P&L | (exit_thesis_ref − entry_thesis_ref) × size, where thesis_ref = the variable the trade was betting on (e.g., breakout follow-through move) | Was the *idea* right? |
| Timing P&L | entry_price vs VWAP of entry day ±1 | Was execution/entry timing the loss source? |
| Sizing error | realized_R × (actual_size − kelly_optimal_size) | Over/under-bet |
| Exit efficiency | (exit_price − best_achievable within hold window) / ATR | Gave back how much? |
| Regime penalty | trade return − average return of same signal in its *favorable* regime | Fired in wrong quadrant? |

Aggregation rule (monthly, per signal): regress realized trade returns on the stored factor_vector.
A factor whose coefficient flips sign vs backtest with t-stat > 2 is flagged **BROKEN-FACTOR** → outer loop
(`loops/outer_loop.py`) proposes threshold retune or retirement. Pre-mortem conditions that fired but
weren't acted on within 1 bar are flagged **DISCIPLINE-FAIL** (bug, not signal problem).

### 9.2 Signal decay detection

Applied to every live signal (implements S5):

| Test | Threshold | Action |
|------|-----------|--------|
| Rolling Sharpe | 90d < 0 AND 252d < 0.3 | Quarantine (S5) |
| CUSUM on daily signal P&L | Two-sided CUSUM, h = 5σ, k = 0.5σ | Alert at drift; quarantine on confirm |
| Hit-rate binomial test | Rolling 50-trade win rate below backtest win rate, p < 0.05 (one-sided binomial) | Halve Kelly fraction |
| Edge half-life | Fit exp decay to 30d-bucketed mean returns since go-live; if half-life < 2 × current holding period | Shorten horizon or retire |
| Crowding proxy | Signal's 90d correlation to generic factor (momentum/carry ETF proxy) rises > 0.8 | Halve weight (edge commoditized) |

Quarantined signals keep paper-trading (`execution/paper_trader.py`); re-admission requires
S3 admission test on post-quarantine data only. Retirement and re-admission events are the
outer loop's training labels.

### 9.3 Regime-conditional strategy weighting

Maintain matrix `PERF[strategy][regime]` = EWMA (half-life 60 trading days) of daily strategy returns,
bucketed by regime state from `macro/engine.py`:

- Regime state = (Dalio quadrant Q1–Q4) × (trend state: ADX(14) > 25 trending / else ranging) × (liquidity D1 on/off) → 16 states.
- Strategy weight_s,r = softmax(PERF[s][r] / temperature) with floor 5% and cap 40% per strategy; temperature annealed by outer loop.
- Hard gates override softmax: CANSLIM "M" rule (distribution-day count ≥ 5 → equity-breakout module weight = 0); D8 circuit breaker (drawdown → all weights scaled); risk engine veto is final.
- Weight updates only on regime-persistence confirmation (R4: 2 consecutive reads) — never intra-day.
- Priors: breakout/trend modules start favored in trending+liquidity-on states; mean-reversion favored in ranging states; Graham/Buffett fundamental sleeve is regime-agnostic but sizes up in Q3/Q4 (cheapness appears in busts).

---

## 10. INTEGRATION BLUEPRINT — Mapping to Engine Modules

Repo: `c:\Users\Asus\Desktop\bitcoin-analyser`. Existing structure → rule assignments:

### 10.1 `screener/fundamental.py` (NEW — sits beside `indicators/structural.py`)
For the equities sleeves (Indian + US):

| Function | Implements | Output |
|----------|-----------|--------|
| `buffett_moat_screen(financials_10y)` | B1–B8, B12, B13 | pass/fail + moat_score 0–8 |
| `owner_earnings(financials)` / `intrinsic_value_dcf()` | B9, B10 | IV per share |
| `margin_of_safety(price, iv)` | B11 | MOS %, buy/strong-buy/reject |
| `graham_netnet_screen(balance_sheet, price)` | G1–G5 | net-net candidate list |
| `graham_defensive_screen()` / `graham_number()` | G6–G13, GR1–GR10 | 7-criteria bitmask + GN |
| `canslim_screen(quarterly, annual, universe_rs)` | CANSLIM C, A, L, I, S | canslim_score 0–7 |
| `management_incentive_filter()` | M1 | binary gate |

Data needed: fundamentals API (screener.in / Tickertape for NSE; FMP/EDGAR for US), 13F data, dividend history.

### 10.2 `indicators/technical.py` + `strategy/modules/breakout.py`
| Function | Implements |
|----------|-----------|
| `trend_template(ohlc, universe)` → 8-bool vector | T1–T8 |
| `detect_vcp(ohlc, vol)` → contractions, pivot, quality score | V1–V6 |
| `cup_handle(ohlc, vol)` → base spec dict, pivot, valid flag | CH1–CH9 |
| `livermore_pivots(ohlc, atr)` → pivotal points, swing filter states | L1–L6, L11 |
| `follow_through_day(index_ohlcv)` / `distribution_days()` | CANSLIM M |
| `reflexivity_divergence(price, fundamental_series)` | D2, D3 |

`breakout.py` consumes VCP/cup-handle pivots (crypto uses the same geometry on BTC/alt bases —
these patterns are asset-agnostic price/volume specs). `trend_follower.py` consumes L1/L4/L13 +
D7. `mean_reverter.py` gated OFF when ADX > 25 (regime matrix 9.3).

### 10.3 `macro/engine.py` — regime detector
| Function | Implements |
|----------|-----------|
| `growth_axis()` / `inflation_axis()` / `quadrant()` | R1–R4 |
| `net_liquidity()` (FRED: WALCL − WTREGEN − RRPONTSYD; + stablecoin mcap for crypto) | D1 |
| `debt_cycle_overlay()` | R5 |
| `cb_pivot_signal()` (2y yield momentum) | D11 |
| `regime_state()` → one of 16 states | feeds 9.3 |

`macro/feeds.py`: add FRED, BIS, breakevens, ISM, claims, stablecoin-supply feeds.

### 10.4 `risk/engine.py` — position sizer + absolute veto (already has veto power per project spec)
| Function | Implements |
|----------|-----------|
| `kelly_fraction(p, b, mu, sigma, corr)` with shrinkage + fractional cap | K1–K6, K8, K9 |
| `pyramid_schedule(full_size)` → [0.2, 0.2, 0.2, 0.4] tranches, add-only-if-higher | L8, D5 |
| VETO list (hard, non-overridable): no averaging down (L9); stop ceiling 7–8% growth-sleeve / 10% absolute (CH10, L10, V7); risk/trade ≤ 1.25% (V8); position caps (V9, K7); reward:risk ≥ 2 (V10), ≥ 3 macro (D4); circuit breakers (D8, D9, R10, V12, V13); cost-basis blindness in hold logic (M3); crowding filter (M4) | — |
| `vol_target()` / `risk_parity_weights(cov)` | R6–R8, S9 |

### 10.5 `strategy/base.py` — signal registry & hygiene
Extend the base class so every module self-reports: signal_id, intended horizon, per-signal P&L
series, live-since date. Registry enforces S1 (≥ 20 signals / ≤ 10% risk each), S6 (correlation
caps), S8 (rationale flag → risk budget multiplier). Order path only through pipeline (S4).

### 10.6 `backtest/engine.py` — admission gate
Implements S3 exactly: t-stat ≥ 3 / PSR ≥ 0.95, walk-forward OOS Sharpe ≥ 0.5 × IS,
survives 2× cost model (S7). No signal reaches `execution/` without a stored admission
certificate (JSON artifact with test results + data hashes) — this is the contract between
research and production.

### 10.7 `loops/` — the Karpathy self-improvement loop
- `loops/inner_loop.py`: per-trade cycle — write TradeRecord at entry (9.1 schema incl. M6 pre-mortem),
  compute attribution decomposition at close, update rolling p/b/μ/σ for Kelly (K9), update
  PERF[strategy][regime].
- `loops/outer_loop.py`: weekly/monthly — run decay tests (9.2), quarantine/retire (S5),
  BROKEN-FACTOR regression (9.1), propose threshold mutations (e.g., widen T8 RS floor 70→75)
  as *experiments*, anneal softmax temperature (9.3).
- `loops/experiment.py`: every proposed mutation becomes an A/B experiment: shadow-run the mutated
  ruleset in paper trading ≥ 60 days or ≥ 30 trades, promote only if it passes the S3 admission
  test against the incumbent. **Thresholds in this spec are mutable ONLY via this path** — never
  hand-edited in production config. Hard risk vetoes in 10.4 are immutable (not searchable by the loop).

### 10.8 `control/` + `dashboard/`
Kill-switch (S4's only manual action), quarantine list, regime state, distribution-day count,
drawdown-breaker status, per-signal rolling Sharpe, `rtk`/attribution reports into `reports/`.

### 10.9 Sleeve architecture summary

| Sleeve | Masters | Modules | Horizon | Sizing |
|--------|---------|---------|---------|--------|
| Quality-value (equities) | Buffett/Munger, Graham defensive | fundamental screener | years | punch-card, B16 |
| Deep-value basket | Graham net-net | fundamental screener | ≤ 2y | equal-weight, G4 |
| Momentum breakout (equities+crypto) | Minervini, O'Neil, Livermore | breakout.py, technical.py | weeks–months | V8/K4 + L8 pyramid |
| Macro directional | Druckenmiller/Soros, Dalio | trend_follower.py, macro/engine.py | weeks–quarters | D5/D6 + K4 |
| Stat/mean-reversion | Simons (hygiene), Thorp | mean_reverter.py + future signals | days | K7 2% caps |
| Allocation layer | Dalio | risk parity across sleeves | monthly | R6–R10 |

### 10.10 Data requirements checklist
OHLCV (all assets, daily + intraday for execution); 10y fundamentals + quarterly (US/India);
dividend & buyback history; 13F/institutional ownership; float/shares; FRED macro (WALCL, WTREGEN,
RRPONTSYD, CPI, claims, GDP nowcast); ISM PMI; 10y breakevens; 2y yields; BCOM; BIS credit/GDP;
crypto: stablecoin supply, funding rates, active addresses/TVL; universe returns for RS ranks;
AAA corporate yields (Graham-Rea); trade-log store (SQLite/parquet) for the learning loop.

---

## Appendix: Immutable vs Learnable parameters

**IMMUTABLE (risk engine, never searched by the loop):** L9 no-averaging-down; CH10/V7 stop ceilings;
V8 risk-per-trade cap; K4 max 0.5-Kelly; D8 circuit breakers; M3 cost-basis blindness; S4 no-override;
G4 net-net diversification floor.

**LEARNABLE (via loops/experiment.py only):** all screen thresholds (T8 RS floor, CANSLIM growth %,
MOS %, quadrant indicator weights), softmax temperature, EWMA half-lives, Kelly shrinkage factor
(within [0.4, 0.6]), holding-period targets, VCP quality weights.
