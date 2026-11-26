"""The shared secret that protects the recommender routes (spec 2026-11-25-book-recommender-design, section 2).

Student ids cross the wire to these routes, and a hosted web service is publicly reachable, so the
Node backend proves itself with the header `x-ai-token`. The token never appears in a log, a response
or an error message: nothing in this module formats it.
"""

import hmac
import os

from fastapi import Header, HTTPException

TOKEN_ENV = "AI_TOKEN"


def require_ai_token(x_ai_token: str | None = Header(default=None)) -> None:
    """FastAPI dependency: 404 while the recommender is off, 401 unless the header matches AI_TOKEN.

    The environment is read on every request, so a token set or rotated at the host takes effect
    without a code change. `AI_TOKEN` unset or blank means the recommender is switched off and the
    routes answer exactly like a route that does not exist. A configured token is compared without
    the whitespace a copied secret often carries; the header is compared as sent. Both are encoded to
    bytes first because `hmac.compare_digest` raises on a str that is not ASCII, which a header can be.
    """
    configured = os.environ.get(TOKEN_ENV, "").strip()
    if not configured:
        raise HTTPException(status_code=404, detail="Not Found")
    if x_ai_token is None or not hmac.compare_digest(
        x_ai_token.encode("utf-8"), configured.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="unauthorized")
