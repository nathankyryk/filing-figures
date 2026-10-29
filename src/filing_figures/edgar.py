"""Polite, cached access to SEC EDGAR.

Three endpoints are all this project needs:
  - submissions:  https://data.sec.gov/submissions/CIK##########.json  (the company's filing list)
  - companyfacts: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json  (every XBRL fact it filed)
  - the archive:  https://www.sec.gov/Archives/edgar/data/<cik>/<accession without dashes>/<document>

SEC fair-access rules: identify yourself (name and email) in the User-Agent and send at most
10 requests a second. We stay under 8. Every response is cached gzipped on disk, so each
file is downloaded once; a re-run reads only the cache.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
from pathlib import Path

import httpx

DEFAULT_CACHE = Path("data/raw")
MAX_PER_SECOND = 8.0
RETRY_STATUS = {429, 500, 502, 503, 504}


def cik10(cik: int | str) -> str:
    """CIK as the zero-padded 10-digit string the data.sec.gov URLs use."""
    return f"{int(cik):010d}"


class RateLimiter:
    """At most `per_second` calls a second, shared across threads."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = self.next_time - now
            self.next_time = max(now, self.next_time) + self.interval
        if delay > 0:
            time.sleep(delay)


class Edgar:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE,
        user_agent: str | None = None,
        per_second: float = MAX_PER_SECOND,
        client: httpx.Client | None = None,
    ):
        user_agent = user_agent or os.environ.get("SEC_USER_AGENT", "")
        if "@" not in user_agent:
            raise ValueError('Set SEC_USER_AGENT to "Your Name your@email" (SEC fair-access rules)')
        self.cache_dir = Path(cache_dir)
        self.limiter = RateLimiter(per_second)
        self.client = client or httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True,
        )

    # -- HTTP -------------------------------------------------------------------------------

    def fetch(self, url: str, tries: int = 6) -> bytes:
        """GET with the rate limit. Retries timeouts, 429 and 5xx with exponential backoff.

        SEC answers 403 both for a missing User-Agent and when a client goes over the rate
        limit (it then blocks for ~10 minutes), so a 403 waits a minute before retrying.
        """
        for attempt in range(tries):
            self.limiter.wait()
            try:
                r = self.client.get(url)
            except httpx.TransportError:
                if attempt == tries - 1:
                    raise
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                r.raise_for_status()
            if attempt == tries - 1:
                r.raise_for_status()
            time.sleep(60 if r.status_code == 403 else 2**attempt)
        raise RuntimeError("unreachable")

    def cached(self, url: str, path: Path) -> bytes:
        """The file at `url`, from the gzipped cache at `path` if present."""
        path = self.cache_dir / path
        if path.exists():
            return gzip.decompress(path.read_bytes())
        data = self.fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(gzip.compress(data, compresslevel=6))
        tmp.replace(path)  # atomic: an interrupted download never leaves a half-written cache file
        return data

    # -- Endpoints ----------------------------------------------------------------------------

    def submissions(self, cik: int | str) -> dict:
        """The filing list, with the older pages merged in when they reach back to 2019.

        `filings.recent` holds at least the last year or 1,000 filings; companies that file
        a lot (debt programmes, insider forms) push older 10-Ks into extra pages.
        """
        c = cik10(cik)
        sub = json.loads(self.cached(f"https://data.sec.gov/submissions/CIK{c}.json", Path(f"submissions/{c}.json.gz")))
        for page in sub["filings"].get("files", []):
            if page.get("filingTo", "") < "2019-01-01":
                continue
            name = page["name"]
            older = json.loads(self.cached(f"https://data.sec.gov/submissions/{name}", Path(f"submissions/{name}.gz")))
            for key, values in older.items():
                sub["filings"]["recent"].setdefault(key, []).extend(values)
        return sub

    def companyfacts(self, cik: int | str) -> dict:
        c = cik10(cik)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c}.json"
        return json.loads(self.cached(url, Path(f"companyfacts/{c}.json.gz")))

    def document(self, cik: int | str, accession: str, name: str) -> bytes:
        """One document of a filing, e.g. the 10-K's primary (inline XBRL) HTML."""
        folder = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{name}"
        return self.cached(url, Path(f"docs/{int(cik)}/{folder}/{name}.gz"))
