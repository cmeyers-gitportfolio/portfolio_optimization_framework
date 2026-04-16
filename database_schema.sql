-- ============================================
-- DATABASE SCHEMA
-- Robust Strategic Asset Allocation Framework
-- ============================================

-- ============================================
-- 1. RAW PRICES
-- ============================================

CREATE TABLE IF NOT EXISTS prices (
    date            DATE NOT NULL,
    ticker          VARCHAR(20) NOT NULL,
    adjusted_close  DOUBLE PRECISION,
    currency        VARCHAR(10),
    PRIMARY KEY (date, ticker)
);

CREATE INDEX idx_prices_ticker ON prices(ticker);
CREATE INDEX idx_prices_date ON prices(date);


-- ============================================
-- 2. FX RATES (CLP/USD)
-- ============================================

CREATE TABLE IF NOT EXISTS fx_rates (
    date        DATE PRIMARY KEY,
    clp_usd     DOUBLE PRECISION
);


-- ============================================
-- 3. RETURNS (USD)
-- ============================================

CREATE TABLE IF NOT EXISTS asset_returns (
    date        DATE NOT NULL,
    ticker      VARCHAR(20) NOT NULL,
    return      DOUBLE PRECISION,
    PRIMARY KEY (date, ticker)
);

CREATE INDEX idx_returns_ticker ON asset_returns(ticker);


-- ============================================
-- 4. RISK-FREE RATE
-- ============================================

CREATE TABLE IF NOT EXISTS risk_free_rate (
    date        DATE PRIMARY KEY,
    rf_monthly  DOUBLE PRECISION
);


-- ============================================
-- 5. PORTFOLIO WEIGHTS (ROBUST MODEL)
-- ============================================

CREATE TABLE IF NOT EXISTS portfolio_weights (
    date        DATE NOT NULL,
    ticker      VARCHAR(20) NOT NULL,
    weight      DOUBLE PRECISION,
    model       VARCHAR(50), -- e.g. 'robust', 'minvar', 'sharpe', 'cvar'
    PRIMARY KEY (date, ticker, model)
);

CREATE INDEX idx_weights_date ON portfolio_weights(date);


-- ============================================
-- 6. BENCHMARK WEIGHTS
-- ============================================

CREATE TABLE IF NOT EXISTS benchmark_weights (
    date        DATE NOT NULL,
    ticker      VARCHAR(20) NOT NULL,
    weight      DOUBLE PRECISION,
    PRIMARY KEY (date, ticker)
);


-- ============================================
-- 7. PORTFOLIO RETURNS
-- ============================================

CREATE TABLE IF NOT EXISTS portfolio_returns (
    date            DATE PRIMARY KEY,
    model_return    DOUBLE PRECISION,
    benchmark_return DOUBLE PRECISION,
    excess_return   DOUBLE PRECISION
);


-- ============================================
-- 8. RISK METRICS (VaR / CVaR)
-- ============================================

CREATE TABLE IF NOT EXISTS risk_metrics (
    date        DATE,
    portfolio   VARCHAR(50), -- 'model', 'benchmark', 'alpha'
    method      VARCHAR(50), -- 'historical', 'normal', 'student_t', etc.
    confidence  DOUBLE PRECISION,
    var         DOUBLE PRECISION,
    cvar        DOUBLE PRECISION
);

CREATE INDEX idx_risk_portfolio ON risk_metrics(portfolio);


-- ============================================
-- 9. MONTE CARLO SIMULATIONS
-- ============================================

CREATE TABLE IF NOT EXISTS monte_carlo_results (
    id              SERIAL PRIMARY KEY,
    simulation_type VARCHAR(50), -- 'normal', 'student_t', 'regime'
    simulated_return DOUBLE PRECISION
);

CREATE INDEX idx_mc_type ON monte_carlo_results(simulation_type);


-- ============================================
-- 10. DISTRIBUTION SUMMARY
-- ============================================

CREATE TABLE IF NOT EXISTS distribution_summary (
    simulation_type VARCHAR(50) PRIMARY KEY,
    mean            DOUBLE PRECISION,
    volatility      DOUBLE PRECISION,
    var_95          DOUBLE PRECISION,
    cvar_95         DOUBLE PRECISION,
    skewness        DOUBLE PRECISION,
    kurtosis        DOUBLE PRECISION
);


-- ============================================
-- 11. STRESS TEST RESULTS
-- ============================================

CREATE TABLE IF NOT EXISTS stress_test_results (
    scenario        VARCHAR(50) PRIMARY KEY,
    portfolio_return DOUBLE PRECISION,
    benchmark_return DOUBLE PRECISION,
    difference      DOUBLE PRECISION
);


-- ============================================
-- 12. ROBUSTNESS RESULTS
-- ============================================

CREATE TABLE IF NOT EXISTS robustness_results (
    id              SERIAL PRIMARY KEY,
    window          INTEGER,
    shrinkage       DOUBLE PRECISION,
    cvar_level      DOUBLE PRECISION,
    return          DOUBLE PRECISION,
    volatility      DOUBLE PRECISION,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    turnover        DOUBLE PRECISION,
    robust_score    DOUBLE PRECISION
);


-- ============================================
-- 13. ESTIMATION ERROR ANALYSIS
-- ============================================

CREATE TABLE IF NOT EXISTS estimation_error_results (
    id                  SERIAL PRIMARY KEY,
    error_type          VARCHAR(50), -- 'returns', 'covariance', 'correlation'
    sharpe_mean         DOUBLE PRECISION,
    sharpe_std          DOUBLE PRECISION,
    drawdown_mean       DOUBLE PRECISION,
    drawdown_std        DOUBLE PRECISION
);


-- ============================================
-- END OF SCHEMA
-- ============================================