I am building an automated stock signal bot.

Goal:
Build a stock trading assistant that can scan stocks, create buy/sell/hold signals, backtest strategies, use news/sentiment, and connect to Alpaca paper trading first. Real/live trading must stay disabled until I explicitly approve it.

Repository:
Review the full GitHub repo, not just this summary. Check every file, especially:
- stock_bot.py
- app.py or Streamlit app files
- requirements.txt
- README.md
- .env.example
- any backtest, news, sentiment, broker, or dashboard files

Current progress:
- Basic stock signal bot exists
- Backtesting exists
- Added or worked on technical indicators
- Added broker safety controls
- Added Alpaca paper trading connection work
- Added environment variable loading with .env
- Added checks for missing Alpaca keys
- Added paper trading mode
- Added order blocking unless broker orders are enabled
- Added UI/status display for Alpaca connection
- Added paper order test section
- Added force test mode for UI testing when no approved order exists
- Added safety rule that real orders should not happen unless explicitly enabled
- Started expanding news/sentiment ideas
- Want news from many sources that could affect stocks, including political posts, IPOs, market news, earnings, economic events, and other catalysts
- Want improvements focused on making money and avoiding losses
- Prefer not to use extra Git branches because previous branch work caused bugs

Important safety rules:
- Do not enable live trading
- Paper trading only
- Real broker orders must stay blocked unless I explicitly approve them
- Do not expose API keys
- Do not commit .env
- Use .env.example for placeholders only
- Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are loaded safely
- ENABLE_BROKER_ORDERS should default to false
- ENABLE_LIVE_ORDERS should default to false

Known issues or concerns:
- Backtest ROI may be unrealistic or overfit
- Need to check for lookahead bias
- Need to check transaction costs, slippage, spread, and realistic fills
- Need to check train/test split
- Need to avoid using future data in signals
- Need better validation before trusting results
- Need clearer project status documentation
- Need better logs and error handling
- Need better paper trading test flow
- Need to confirm all environment variables work correctly
- Need to clean duplicate files if there are two versions of the same thing

Ask Codex to do this:
1. Review the entire repo
2. Explain the current stage of the project
3. Find bugs or fragile code
4. Check whether the backtest is realistic
5. Look for lookahead bias
6. Improve risk management
7. Improve the signal quality
8. Improve news and sentiment handling
9. Improve Alpaca paper trading safety
10. Improve project structure
11. Add or update README.md
12. Add PROJECT_STATUS.md
13. Add .env.example if missing
14. Add tests where useful
15. Tell me what to build next

Risk management features I want:
- Position sizing
- Stop loss
- Take profit
- Max daily loss
- Max position count
- Max allocation per trade
- Cooldown after losses
- Avoid trading during extreme volatility
- Paper trading before live trading
- Manual approval before live orders

News and catalyst features I want:
- Earnings dates
- Analyst upgrades/downgrades
- IPOs
- SEC filings
- Major economic data
- Federal Reserve news
- Political or policy news that affects markets
- Company specific news
- Sentiment scoring
- Avoid fake or low quality news if possible

Training/backtesting features I want:
- Walk forward testing
- Train/test split
- Out of sample testing
- Slippage and commission assumptions
- Compare against buy and hold
- Track win rate, max drawdown, Sharpe ratio, profit factor, and total return
- Avoid overfitting
- Save backtest results
- Show clear charts and metrics

Prompt for Codex:
“Review this repo as if you are a senior quant developer and trading systems engineer. Tell me exactly what has been built, what is broken, what is risky, and what I should improve next. Focus on realistic profitability, avoiding losses, safe paper trading, clean code, and preventing fake backtest results. Do not enable live trading. Make changes only if they are safe and explain each change.”