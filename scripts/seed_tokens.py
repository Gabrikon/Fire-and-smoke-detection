"""Seed access tokens for the fire/smoke app.

Idempotent: ensures the `fire.access_tokens` table exists and, if it is empty, seeds
3 primary + 7 secondary tokens and prints them. Safe to run repeatedly.

Usage:
    python scripts/seed_tokens.py            # reads SUPABASE_DB_URL / DB_SCHEMA from .env or env
"""
from __future__ import annotations

import asyncio
import os
import secrets

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

import asyncpg

DB_URL = os.environ.get("SUPABASE_DB_URL", "")
SCHEMA = os.environ.get("DB_SCHEMA", "fire")
N_PRIMARY = 3
N_SECONDARY = 7


async def _connect():
    for ssl_opt in (False, True):
        try:
            return await asyncpg.connect(dsn=DB_URL, ssl=ssl_opt)
        except Exception:  # noqa: BLE001
            continue
    raise SystemExit("Could not connect. Check SUPABASE_DB_URL.")


async def main() -> None:
    if not DB_URL:
        raise SystemExit("SUPABASE_DB_URL is not set (see .env.example).")

    conn = await _connect()
    try:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        await conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {SCHEMA}.access_tokens (
                    id BIGSERIAL PRIMARY KEY,
                    token TEXT UNIQUE NOT NULL,
                    label TEXT,
                    kind TEXT DEFAULT 'secondary',
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT now())"""
        )
        count = await conn.fetchval(f"SELECT count(*) FROM {SCHEMA}.access_tokens")
        if count and count > 0:
            print(f"access_tokens already has {count} rows; not seeding.")
            rows = await conn.fetch(
                f"SELECT token, kind, active FROM {SCHEMA}.access_tokens ORDER BY kind, id"
            )
            for r in rows:
                print(f"  [{r['kind']}] {r['token']}  active={r['active']}")
            return

        created = []
        for i in range(N_PRIMARY):
            tok = f"fire-pri-{secrets.token_hex(8)}"
            await conn.execute(
                f"INSERT INTO {SCHEMA}.access_tokens (token, label, kind) VALUES ($1,$2,'primary')",
                tok, f"primary-{i+1}",
            )
            created.append(("primary", tok))
        for i in range(N_SECONDARY):
            tok = f"fire-sec-{secrets.token_hex(8)}"
            await conn.execute(
                f"INSERT INTO {SCHEMA}.access_tokens (token, label, kind) VALUES ($1,$2,'secondary')",
                tok, f"secondary-{i+1}",
            )
            created.append(("secondary", tok))

        print(f"Seeded {len(created)} tokens into {SCHEMA}.access_tokens:")
        for kind, tok in created:
            print(f"  [{kind}] {tok}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
