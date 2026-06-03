CREATE TABLE observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id   INTEGER NOT NULL REFERENCES stations(id),
    parameter_id INTEGER NOT NULL REFERENCES parameters(id),
    datetime     TEXT    NOT NULL,
    value        REAL,
    quality      TEXT    DEFAULT 'measured',
    UNIQUE(station_id, parameter_id, datetime)
);

CREATE TABLE parameters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    unit        TEXT    NOT NULL,
    description TEXT
);

CREATE TABLE risk_sites (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT    NOT NULL UNIQUE,
    lon                    REAL    NOT NULL,
    lat                    REAL    NOT NULL,
    nearest_station        TEXT    REFERENCES stations(code),
    risk_class             INTEGER,
    risk_class_name        TEXT,
    retreat_rate_m_per_yr  REAL,
    issue_type             TEXT,
    severity               TEXT
);

CREATE TABLE stations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    code    TEXT    NOT NULL UNIQUE,
    name    TEXT    NOT NULL,
    lon     REAL    NOT NULL,
    lat     REAL    NOT NULL,
    type    TEXT    DEFAULT 'tidal'
);

CREATE INDEX idx_obs_station   ON observations(station_id);

CREATE INDEX idx_obs_parameter ON observations(parameter_id);

CREATE INDEX idx_obs_datetime  ON observations(datetime);

CREATE INDEX idx_obs_combo     ON observations(station_id, parameter_id, datetime);

CREATE VIEW ssc_risk_summary AS
SELECT
    rs.name                          AS site_name,
    rs.risk_class,
    rs.risk_class_name,
    rs.issue_type,
    rs.severity,
    st.code                          AS station,
    st.name                          AS station_name,
    ROUND(AVG(o.value), 2)           AS mean_ssc_mg_l,
    ROUND(MAX(o.value), 2)           AS max_ssc_mg_l,
    ROUND(MIN(o.value), 2)           AS min_ssc_mg_l,
    COUNT(o.id)                      AS n_obs,
    MIN(o.datetime)                  AS period_start,
    MAX(o.datetime)                  AS period_end
FROM risk_sites rs
JOIN stations   st ON st.code = rs.nearest_station
JOIN observations o ON o.station_id = st.id
JOIN parameters  p  ON p.id = o.parameter_id
WHERE p.code = 'CONCTTE'
GROUP BY rs.name, st.code;

CREATE VIEW monthly_means AS
SELECT
    st.code                          AS station,
    p.code                           AS parameter,
    p.unit,
    SUBSTR(o.datetime, 1, 7)         AS year_month,
    ROUND(AVG(o.value), 4)           AS mean_value,
    ROUND(MIN(o.value), 4)           AS min_value,
    ROUND(MAX(o.value), 4)           AS max_value,
    COUNT(o.id)                      AS n_obs
FROM observations o
JOIN stations   st ON st.id = o.station_id
JOIN parameters p  ON p.id  = o.parameter_id
GROUP BY st.code, p.code, year_month;

CREATE VIEW annual_trends AS
SELECT
    st.code                          AS station,
    p.code                           AS parameter,
    p.unit,
    SUBSTR(o.datetime, 1, 4)         AS year,
    ROUND(AVG(o.value), 4)           AS annual_mean,
    COUNT(o.id)                      AS n_obs
FROM observations o
JOIN stations   st ON st.id = o.station_id
JOIN parameters p  ON p.id  = o.parameter_id
GROUP BY st.code, p.code, year;

