# WhatsApp → Upstox Approved Execution

A human-in-the-loop service for turning trading-group messages into validated trade candidates, requiring explicit approval before any broker order. This package is intentionally paper-only in its initial slice: no live order-placement endpoint is implemented or enabled.

## Safety status

- Live trading: **NOT IMPLEMENTED / DISABLED**.
- Broker adapter: deterministic simulator only in this first slice.
- All approvals are one-time, expiring, and bound to an immutable order payload hash.
- Missing contract expiry, stale quote, missing risk limits, duplicates, and ambiguous signals fail closed.

## Quick start

```bash
cd whatsapp_upstox
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the local API docs.

## Initial workflow

1. `POST /v1/messages` captures signal text; event-key idempotency prevents repeated ingestion.
2. `POST /v1/messages/{message_id}/parse` extracts a conservative structured trade candidate.
3. `POST /v1/candidates/{candidate_id}/validate` applies configured risk limits and requires fully resolved contract metadata.
4. `POST /v1/candidates/{candidate_id}/approval` creates a short-lived approval bound to the exact order payload.
5. `POST /v1/approvals/{approval_id}/approve` consumes the approval exactly once, then submits to the simulated broker only.
6. `GET /v1/audit` returns audit events.

The parser in this starter supports clear text messages only. Image/OCR/audio pipelines and Android capture are next milestones; do not pretend notification access reliably provides original group media.

## Configuration

Risk thresholds are configured through environment variables. The service intentionally starts with **live execution disabled** and broker mode `simulated`. Do not expose it publicly or use real trading until authentication, secret storage, Upstox requirements, static-IP registration, explicit limits, and comprehensive tests are complete.
