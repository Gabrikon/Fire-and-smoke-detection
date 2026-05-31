-- Fire/smoke detection schema, isolated from the naija-petro project under its own `fire_detection` schema.
-- Run once against your self-hosted Supabase / Postgres (Railway):
--   psql "$SUPABASE_DB_URL" -f supabase/schema.sql

CREATE SCHEMA IF NOT EXISTS fire_detection;

-- Visitors (open UI; email captured optionally). "Store the users."
CREATE TABLE IF NOT EXISTS fire_detection.users (
    id          BIGSERIAL PRIMARY KEY,
    email       TEXT UNIQUE,
    ip_hash     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    last_seen   TIMESTAMPTZ DEFAULT now()
);

-- Optional camera/site labels for a feed.
CREATE TABLE IF NOT EXISTS fire_detection.sites (
    id          BIGSERIAL PRIMARY KEY,
    label       TEXT,
    location    TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- One row per analyzed frame/clip that produced detections.
CREATE TABLE IF NOT EXISTS fire_detection.detection_events (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ DEFAULT now(),
    user_id       BIGINT REFERENCES fire_detection.users(id),
    ip_hash       TEXT,
    source        TEXT,                 -- webcam | image | video
    site_id       BIGINT REFERENCES fire_detection.sites(id),
    n_detections  INT DEFAULT 0,
    alert_level   TEXT,                 -- CLEAR | LOW | MEDIUM | HIGH
    detections    JSONB                 -- [{class, confidence, bbox}]
);

-- The structured safety advisory for an event (also serves as alert history for the UI).
CREATE TABLE IF NOT EXISTS fire_detection.advisories (
    id                  BIGSERIAL PRIMARY KEY,
    created_at          TIMESTAMPTZ DEFAULT now(),
    event_id            BIGINT REFERENCES fire_detection.detection_events(id) ON DELETE CASCADE,
    severity            TEXT,
    threat_type         TEXT,
    is_false_alarm      BOOLEAN,
    estimated_scale     TEXT,
    escalation_level    TEXT,
    confidence          REAL,
    recommended_actions JSONB,
    dam_description      TEXT,           -- localized description from NVIDIA DAM-3B
    model_used          TEXT,           -- NIM reasoning model id
    latency_ms          INT,
    raw                 JSONB
);

-- Access tokens (open UI bypasses the daily free limit when a valid token is provided).
CREATE TABLE IF NOT EXISTS fire_detection.access_tokens (
    id          BIGSERIAL PRIMARY KEY,
    token       TEXT UNIQUE NOT NULL,
    label       TEXT,
    kind        TEXT DEFAULT 'secondary',   -- primary | secondary
    active      BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Usage log (per-IP daily free-limit counting + admin analytics).
CREATE TABLE IF NOT EXISTS fire_detection.usage_events (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    ip_hash     TEXT,
    kind        TEXT,                       -- detect | advise | advise_video
    user_id     BIGINT,
    token_used  TEXT
);

CREATE INDEX IF NOT EXISTS idx_fire_events_created ON fire_detection.detection_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fire_adv_event      ON fire_detection.advisories(event_id);
CREATE INDEX IF NOT EXISTS idx_fire_adv_created    ON fire_detection.advisories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fire_usage_created  ON fire_detection.usage_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fire_usage_ip       ON fire_detection.usage_events(ip_hash);
