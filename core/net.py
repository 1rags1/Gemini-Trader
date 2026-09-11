"""Networking prerequisites for outbound HTTPS.

Python's HTTP stack validates certificates against the `certifi` bundle, which
does not include locally-installed root CAs. On machines where antivirus or a
corporate proxy terminates TLS (Norton, Kaspersky, Zscaler, ...), every Gemini
request fails with CERTIFICATE_VERIFY_FAILED even though the key is valid.

`truststore` redirects verification to the OS certificate store, where that root
CA is already trusted. Verification stays fully enabled.
"""

from __future__ import annotations

_injected = False


def enable_os_trust_store() -> bool:
    """Route TLS verification through the OS trust store. Idempotent.

    Returns True if injection is active. Must be called before any HTTPS client
    is constructed, since it patches `ssl.SSLContext` at the module level.
    """
    global _injected
    if _injected:
        return True

    try:
        import truststore
    except ImportError:
        return False

    truststore.inject_into_ssl()
    _injected = True
    return True
