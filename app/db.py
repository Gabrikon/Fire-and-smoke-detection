"""Async Postgres layer (self-hosted Supabase on Railway).

Stores users, detection events, advisories, access tokens, and usage. Everything degrades
gracefully if SUPABASE_DB_URL is not set: the open UI keeps working, just without persistence or
the per-IP daily limit.

Connection follows the naija-petro pattern: the Railway public proxy speaks plaintext, so we try a
non-SSL connection first and fall back to SSL.
"""
from __future__ import annotations

import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)

SCHEMA = settings.db_schema
_pool = None
_pool_tried = False


async def _get_pool():
    """Lazily create a connection pool, or return None if the DB is unreachable/unconfigured."""
    global _pool, _pool_tried
    if _pool is not None:
        return _pool
    if _pool_tried or not settings.db_url:
        return None
    _pool_tried = True

    import asyncpg

    for ssl_opt in (False, True):
        try:
            _pool = await asyncpg.create_pool(
                dsn=settings.db_url,
                ssl=ssl_opt,
                min_size=1,
                max_size=5,
                command_timeout=30,
                server_settings={"search_path": f"{SCHEMA},public"},
            )
            logger.info("DB pool created (ssl=%s, schema=%s)", ssl_opt, SCHEMA)
            return _pool
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB pool ssl=%s failed: %s", ssl_opt, exc)
    logger.error("Could not connect to the database; persistence disabled.")
    return None


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
async def upsert_user(email: str | None, ip_hash: str | None) -> int | None:
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        if email:
            row = await conn.fetchrow(
                f"""INSERT INTO {SCHEMA}.users (email, ip_hash)
                    VALUES ($1, $2)
                    ON CONFLICT (email) DO UPDATE SET last_seen = now(), ip_hash = EXCLUDED.ip_hash
                    RETURNING id""",
                email, ip_hash,
            )
        else:
            row = await conn.fetchrow(
                f"INSERT INTO {SCHEMA}.users (ip_hash) VALUES ($1) RETURNING id", ip_hash,
            )
        return row["id"] if row else None


async def log_event(*, user_id: int | None, ip_hash: str | None, source: str,
                    detections: list[dict], alert_level: str) -> int | None:
    pool = await _get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""INSERT INTO {SCHEMA}.detection_events
                (user_id, ip_hash, source, n_detections, alert_level, detections)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb) RETURNING id""",
            user_id, ip_hash, source, len(detections), alert_level, json.dumps(detections),
        )
        return row["id"] if row else None


async def log_advisory(*, event_id: int | None, advisory: dict, dam_description: str,
                       model_used: str, latency_ms: int) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            f"""INSERT INTO {SCHEMA}.advisories
                (event_id, severity, threat_type, is_false_alarm, estimated_scale,
                 escalation_level, confidence, recommended_actions, dam_description,
                 model_used, latency_ms, raw)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10,$11,$12::jsonb)""",
            event_id,
            advisory.get("severity"),
            advisory.get("threat_type"),
            bool(advisory.get("is_false_alarm", False)),
            advisory.get("estimated_scale"),
            advisory.get("escalation_level"),
            float(advisory.get("confidence", 0) or 0),
            json.dumps(advisory.get("recommended_actions", [])),
            dam_description,
            model_used,
            int(latency_ms),
            json.dumps(advisory),
        )


async def log_usage(*, ip_hash: str | None, kind: str, user_id: int | None = None,
                    token_used: str | None = None) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            f"INSERT INTO {SCHEMA}.usage_events (ip_hash, kind, user_id, token_used) "
            f"VALUES ($1,$2,$3,$4)",
            ip_hash, kind, user_id, token_used,
        )


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
async def daily_ip_count(ip_hash: str | None) -> int:
    """Number of VLM advisories this IP has triggered today (UTC). 0 if no DB."""
    pool = await _get_pool()
    if pool is None or not ip_hash:
        return 0
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            f"""SELECT count(*) FROM {SCHEMA}.usage_events
                WHERE ip_hash = $1 AND kind IN ('advise','advise_video')
                AND created_at >= date_trunc('day', now())""",
            ip_hash,
        )
        return int(val or 0)


async def token_active(token: str | None) -> bool:
    pool = await _get_pool()
    if pool is None or not token:
        return False
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            f"SELECT active FROM {SCHEMA}.access_tokens WHERE token = $1", token,
        )
        return bool(val)


async def recent_advisories(limit: int = 20) -> list[dict]:
    pool = await _get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""SELECT a.created_at, a.severity, a.threat_type, a.escalation_level,
                       a.confidence, e.source
                FROM {SCHEMA}.advisories a
                LEFT JOIN {SCHEMA}.detection_events e ON e.id = a.event_id
                ORDER BY a.created_at DESC LIMIT $1""",
            limit,
        )
        return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #
async def list_tokens() -> list[dict]:
    pool = await _get_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT id, token, label, kind, active, created_at "
            f"FROM {SCHEMA}.access_tokens ORDER BY kind, id",
        )
        return [dict(r) for r in rows]


async def set_token_active(token_id: int, active: bool) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE {SCHEMA}.access_tokens SET active = $2 WHERE id = $1", token_id, active,
        )


async def count_tokens_by_kind(kind: str) -> int:
    pool = await _get_pool()
    if pool is None:
        return 0
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            f"SELECT count(*) FROM {SCHEMA}.access_tokens WHERE kind = $1", kind,
        )
        return int(val or 0)


async def create_token(token: str, label: str, kind: str) -> None:
    pool = await _get_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            f"INSERT INTO {SCHEMA}.access_tokens (token, label, kind) VALUES ($1,$2,$3)",
            token, label, kind,
        )


async def usage_overview(days: int = 14) -> dict:
    """Summary stats for the admin panel."""
    pool = await _get_pool()
    if pool is None:
        return {"summary": {}, "today": {}, "daily": []}
    async with pool.acquire() as conn:
        summary = await conn.fetchrow(
            f"""SELECT
                  (SELECT count(*) FROM {SCHEMA}.users) AS users,
                  (SELECT count(*) FROM {SCHEMA}.detection_events) AS events,
                  (SELECT count(*) FROM {SCHEMA}.advisories) AS advisories""",
        )
        today = await conn.fetchrow(
            f"""SELECT
                  (SELECT count(*) FROM {SCHEMA}.detection_events
                     WHERE created_at >= date_trunc('day', now())) AS events,
                  (SELECT count(*) FROM {SCHEMA}.advisories
                     WHERE created_at >= date_trunc('day', now())) AS advisories""",
        )
        daily = await conn.fetch(
            f"""SELECT date_trunc('day', created_at)::date AS day, count(*) AS advisories
                FROM {SCHEMA}.advisories
                WHERE created_at >= now() - ($1 || ' days')::interval
                GROUP BY 1 ORDER BY 1 DESC""",
            days,
        )
        return {
            "summary": dict(summary) if summary else {},
            "today": dict(today) if today else {},
            "daily": [dict(r) for r in daily],
        }
