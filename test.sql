-- =========================================================
-- PNL DATABASE - CLEAN PRODUCTION SETUP
-- PostgreSQL + TimescaleDB
-- =========================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =========================================================
-- DIMENSION TABLES
-- =========================================================

CREATE TABLE IF NOT EXISTS dim_desk (
    desk_id SMALLSERIAL PRIMARY KEY,
    desk_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_portfolio (
    portfolio_id SERIAL PRIMARY KEY,
    portfolio_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_pnl_type (
    pnl_type_id SMALLSERIAL PRIMARY KEY,
    pnl_type_name TEXT UNIQUE NOT NULL
);

-- =========================================================
-- FACT TABLE
-- =========================================================

CREATE TABLE IF NOT EXISTS pnl_timeseries (
    ts              TIMESTAMPTZ NOT NULL,
    desk_id         SMALLINT NOT NULL,
    portfolio_id    INT NOT NULL,
    pnl_type_id     SMALLINT NOT NULL,
    pnl_value       DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (
        ts,
        desk_id,
        portfolio_id,
        pnl_type_id
    ),

    FOREIGN KEY (desk_id)
        REFERENCES dim_desk(desk_id),

    FOREIGN KEY (portfolio_id)
        REFERENCES dim_portfolio(portfolio_id),

    FOREIGN KEY (pnl_type_id)
        REFERENCES dim_pnl_type(pnl_type_id)
);

-- =========================================================
-- INDEXES
-- =========================================================

CREATE INDEX IF NOT EXISTS idx_pnl_desk_ts
ON pnl_timeseries (desk_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_pnl_portfolio_ts
ON pnl_timeseries (portfolio_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_pnl_type_ts
ON pnl_timeseries (pnl_type_id, ts DESC);

-- =========================================================
-- HYPERTABLE
-- =========================================================

SELECT create_hypertable(
    'pnl_timeseries',
    'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =========================================================
-- COMPRESSION
-- =========================================================

ALTER TABLE pnl_timeseries SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'desk_id, portfolio_id, pnl_type_id',
    timescaledb.compress_orderby = 'ts DESC'
);

SELECT add_compression_policy(
    'pnl_timeseries',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- =========================================================
-- -- RETENTION (optional: 2 years)
-- -- =========================================================

-- SELECT add_retention_policy(
--     'pnl_timeseries',
--     INTERVAL '2 years',
--     if_not_exists => TRUE
-- );

-- =========================================================
-- END OF DAY CONTINUOUS AGGREGATE
-- =========================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS pnl_eod
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', ts) AS day,
    desk_id,
    portfolio_id,
    pnl_type_id,
    last(pnl_value, ts) AS pnl_value
FROM pnl_timeseries
GROUP BY 1,2,3,4
WITH NO DATA;

-- =========================================================
-- AUTO REFRESH POLICY
-- =========================================================

SELECT add_continuous_aggregate_policy(
    'pnl_eod',
    start_offset => INTERVAL '30 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '15 minutes',
    if_not_exists => TRUE
);

-- =========================================================
-- INITIAL REFRESH
-- =========================================================

CALL refresh_continuous_aggregate(
    'pnl_eod',
    NULL,
    NULL
);

-- =========================================================
-- USEFUL QUERIES
-- =========================================================

-- Latest snapshot
-- SELECT DISTINCT ON (desk_id, portfolio_id, pnl_type_id)
--     ts, desk_id, portfolio_id, pnl_type_id, pnl_value
-- FROM pnl_timeseries
-- ORDER BY desk_id, portfolio_id, pnl_type_id, ts DESC;

-- Grafana intraday
-- SELECT
--   time_bucket($__interval, ts) AS time,
--   SUM(pnl_value) AS pnl
-- FROM pnl_timeseries
-- WHERE ts BETWEEN $__timeFrom() AND $__timeTo()
-- GROUP BY 1
-- ORDER BY 1;

-- Historical EOD
-- SELECT * FROM pnl_eod ORDER BY day;


-- CREATE VIEW pnl_timeseries_readable AS
-- SELECT
--     p.ts,
--     d.desk_name,
--     pf.portfolio_name,
--     t.pnl_type_name,
--     p.pnl_value
-- FROM pnl_timeseries p
-- JOIN dim_desk d
--     ON p.desk_id = d.desk_id
-- JOIN dim_portfolio pf
--     ON p.portfolio_id = pf.portfolio_id
-- JOIN dim_pnl_type t
--     ON p.pnl_type_id = t.pnl_type_id;


-- CREATE VIEW pnl_eod_readable AS
-- SELECT
--     e.day,
--     d.desk_name,
--     pf.portfolio_name,
--     t.pnl_type_name,
--     e.pnl_value
-- FROM pnl_eod e
-- JOIN dim_desk d ON e.desk_id = d.desk_id
-- JOIN dim_portfolio pf ON e.portfolio_id = pf.portfolio_id
-- JOIN dim_pnl_type t ON e.pnl_type_id = t.pnl_type_id;

-- grafana intraday with filters 2 variables
-- SELECT desk_id, desk_name FROM dim_desk
-- SELECT
--   time_bucket($__interval, ts) AS time,
--   SUM(pnl_value) AS pnl
-- FROM pnl_timeseries
-- WHERE ts BETWEEN $__timeFrom() AND $__timeTo()
--   AND desk_id IN (${desk:csv})
--   AND portfolio_id IN (${portfolio:csv})
-- GROUP BY 1
-- ORDER BY 1;

-- SELECT *
-- FROM pnl_timeseries_readable
-- LIMIT 10;
