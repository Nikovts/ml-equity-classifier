# Multi-Factor Equity Classification Framework: Fusing Fundamentals, Technicals, and Macro Risk

## 1. Problem Formulation & Significance
Predicting short-term equity price movements is notoriously difficult due to high market efficiency and a low signal-to-noise ratio. This project pivots away from short-term "noise" to focus on a medium-term **3-to-6-month investment horizon**, where fundamental value and structural trends exert greater influence. 

The goal is to build a multi-factor machine learning classifier that predicts whether a specific stock will **Outperform (1)**, **Match (0)**, or **Underperform (-1)** the benchmark index (S&P 500). 

### Mathematical Target Definition
Let $R_{\text{asset}}$ be the log return of the target asset over the forward horizon $T$ (3–6 months), and $R_{\text{bench}}$ be the log return of the S&P 500 index over the same period. The target label $Y$ is formally defined as:

$$Y = \begin{cases} 1 & \text{if } R_{\text{asset}} - R_{\text{bench}} > \alpha \\ 0 & \text{if } -\alpha \le R_{\text{asset}} - R_{\text{bench}} \le \alpha \\ -1 & \text{if } R_{\text{asset}} - R_{\text{bench}} < -\alpha \end{cases}$$

*(Where $\alpha = 0.05$, representing a 5% outperformance/underperformance threshold relative to the benchmark).*

This approach is highly significant for systematic asset management, helping automate equity selection using verifiable, data-driven mathematical models rather than intuition.

---

## 2. Project Scope & Features
The model processes four distinct dimensions of market information:
1. **Fundamental Factors:** Intrinsic valuation metrics derived from an automated Discounted Cash Flow (DCF) model alongside comparative valuation ratios (P/E, P/B, EV/EBITDA) normalized against industry averages.
2. **Technical Factors:** Trend, Momentum, Volatility, and Volume tracking indicators (e.g., MACD, RSI, Bollinger Bands, ATR).
3. **Macro Risk & Volatility:** Broader systemic risk modeling utilizing historical Implied Volatility indices (VIX).
4. **Alternative Data:** Sentiment scores generated from corporate news streams.

---

## 3. Known Constraints & Assumptions
* **Look-Ahead Bias Mitigation:** Fundamental accounting metrics are subject to a strict 3-month reporting lag constraint to simulate real-world data availability (e.g., Q3 financials cannot be utilized by the model until their actual public release date).
* **Market Efficiency Assumption:** We assume a baseline directional accuracy hurdle rate of 53%–55% as a statistically significant market edge, prioritizing positive mathematical expectancy and asymmetric risk-reward metrics over raw accuracy.
* **Transaction Costs:** Backtesting engines assume a fixed execution drag (slippage and fees) to preserve real-world utility.

---

## 4. Evaluation Metrics
Models will be thoroughly evaluated beyond standard classification matrices:
* **Machine Learning Metrics:** Precision (specifically for the Buy class), F1-Score, and Cohen's Kappa.
* **Financial Backtesting Metrics:** Sharpe Ratio, Maximum Drawdown, and Information Ratio.
