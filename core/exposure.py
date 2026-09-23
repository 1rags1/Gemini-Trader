"""Localhost-only defaults for the webhook and the dashboard.

An empty shared secret is acceptable on loopback. Binding any other host, or
accepting a tunneled / non-local request, requires a secret.
"""

from __future__ import annotations

from typing import Any

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

#: Headers Cloudflare/cloudflared attach. A laptop browser on 127.0.0.1 does
#: not send these; a phone hitting a tunnel does, even if the peer is local.
TUNNEL_HEADERS = ("cf-ray", "cf-connecting-ip", "cdn-loop")


def is_loopback_host(host: str | None) -> bool:
    """True for localhost bind addresses and loopback client hosts."""
    if not host:
        return False
    normalized = host.strip().lower()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized.count(":") == 1 and "." in normalized:
        normalized = normalized.split(":", 1)[0]
    return normalized in LOOPBACK_HOSTS


def is_tunneled(headers: Any) -> bool:
    """True when the request carries a tunnel/CDN header."""
    if headers is None:
        return False
    try:
        pairs = list(headers.items())
    except Exception:
        return False
    present = {str(key).lower() for key, value in pairs if value}
    return any(name in present for name in TUNNEL_HEADERS)


def request_from_localhost(request: Any) -> bool:
    """True only for a loopback peer that is not arriving through a tunnel."""
    if is_tunneled(getattr(request, "headers", None)):
        return False
    client = getattr(request, "client", None)
    host = getattr(client, "host", "") if client is not None else ""
    return is_loopback_host(host or "")


def assert_secret_for_public_bind(
    *,
    host: str,
    secret: str,
    service: str,
    secret_name: str,
) -> None:
    """Refuse to listen beyond localhost without a shared secret."""
    if is_loopback_host(host) or (secret or "").strip():
        return
    raise RuntimeError(
        f"{secret_name} is required when {service} binds to {host!r}. "
        "An empty secret is only allowed for localhost. "
        f"Set {secret_name}, or bind to 127.0.0.1."
    )
