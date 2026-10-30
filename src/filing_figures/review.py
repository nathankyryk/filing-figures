"""The hand-check page: check the test split's answer key by eye, one filing at a time.

`figures review` serves it on 127.0.0.1. Each filing shows its six gold values next to the input
lines they came from, the other tagged values for the same period, and a link to the filing on
EDGAR. Verdicts go to labels/handcheck.csv (see `dataset.apply_handcheck`); `figures export`
applies them.

The pool (labels/handcheck_pool.csv, drawn once and kept): a seeded random sample of test filings,
whose fix rate is an unbiased estimate of the answer key's error rate, plus the filings with a flag
that can change a score, most important first, up to `target` filings.
"""

from __future__ import annotations

import csv
import json
import random
import re
import threading
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from filing_figures.dataset import VERDICTS, read_handcheck, write_handcheck
from filing_figures.evaluate import close
from filing_figures.labels import FIELD_NAMES, FIELDS, MAX_YEAR_DAYS, MIN_YEAR_DAYS
from filing_figures.schema import SCALE_POWER, SCALES

PORT = 7862
POOL_COLUMNS = ("order", "id", "reason")
# Flags worth a look, most important first. multi_tag and tag_from_statement count only when the
# other tag's value would score differently (e.g. net income including noncontrolling interests).
REVIEW_FLAGS = ("conflict", "mismatch", "date_from_text", "tag_from_statement", "multi_tag")
KIND = {f.name: f.kind for f in FIELDS}
OWN_TAGS = {f.name: f.tags for f in FIELDS}
RELATED = {  # tags shown next to each gold value, for the same period
    **OWN_TAGS,
    "net_income": (
        "us-gaap:NetIncomeLoss",
        "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
        "us-gaap:ProfitLoss",
        "us-gaap:NetIncomeLossAttributableToNoncontrollingInterest",
        "us-gaap:IncomeLossFromContinuingOperations",
    ),
    "eps_diluted": (
        "us-gaap:EarningsPerShareDiluted",
        "us-gaap:EarningsPerShareBasicAndDiluted",
        "us-gaap:EarningsPerShareBasic",
    ),
    "shares_outstanding": ("dei:EntityCommonStockSharesOutstanding", "us-gaap:CommonStockSharesOutstanding"),
}
MAX_BODY = 64_000
MAX_NOTE = 500


def edgar_url(rec: dict) -> str:
    """The filing in EDGAR's inline XBRL viewer (shows each tagged fact on click)."""
    acc = rec["accession"].replace("-", "")
    return f"https://www.sec.gov/ix?doc=/Archives/edgar/data/{rec['cik']}/{acc}/{rec['primary_document']}"


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days + 1


def related_facts(rec: dict, name: str) -> list[dict]:
    """Facts of the field's related tags for the filing's current period (shares: on or after it)."""
    period_end, out, seen = rec["period_end"], [], set()
    for f in rec["context_facts"]:
        if f["tag"] not in RELATED[name]:
            continue
        if KIND[name] == "year":
            ok = f["start"] and f["end"] == period_end and MIN_YEAR_DAYS <= _days(f["start"], f["end"]) <= MAX_YEAR_DAYS
        elif KIND[name] == "instant":
            ok = not f["start"] and f["end"] == period_end
        else:
            ok = not f["start"] and f["end"] >= period_end
        key = (f["tag"], f["end"], f["val"])
        if ok and key not in seen:
            seen.add(key)
            out.append({"tag": f["tag"], "value": f["val"], "start": f["start"], "end": f["end"]})
    order = {t: i for i, t in enumerate(RELATED[name])}
    return sorted(out, key=lambda f: (order[f["tag"]], f["end"]))


def review_reasons(rec: dict) -> list[str]:
    """ "field:flag" for each flag that can change this filing's score."""
    out = []
    for name in FIELD_NAMES:
        gold, flags = rec["gold"][name], rec["flags"][name]
        for flag in REVIEW_FLAGS:
            if flag not in flags:
                continue
            if flag in ("multi_tag", "tag_from_statement"):
                others = [f["value"] for f in related_facts(rec, name) if f["tag"] in OWN_TAGS[name]]
                if not gold or all(close(v, gold["value"], name) for v in others):
                    continue
            out.append(f"{name}:{flag}")
    return out


def _rank(reasons: list[str]) -> int:
    flags = [r.split(":")[1] for r in reasons]
    if any(f != "multi_tag" for f in flags):
        return 0
    if any(r != "net_income:multi_tag" for r in reasons):
        return 1
    return 2


def build_pool(records: list[dict], target: int = 100, n_random: int = 40, seed: int = 0) -> list[dict]:
    """[{order, id, reason}]: n_random seeded random filings, then flagged ones, most important first."""
    rng = random.Random(seed)
    ids = sorted(r["id"] for r in records)
    reasons = {r["id"]: review_reasons(r) for r in records}
    picked = rng.sample(ids, min(n_random, len(ids)))
    rows = [(i, ";".join(["random", *reasons[i]])) for i in picked]
    rest = [i for i in ids if i not in set(picked) and reasons[i]]
    rng.shuffle(rest)
    rest.sort(key=lambda i: _rank(reasons[i]))  # stable: shuffled within a rank
    rows += [(i, ";".join(reasons[i])) for i in rest[: max(0, target - len(rows))]]
    rng.shuffle(rows)  # random and flagged mixed, so tiredness doesn't fall on one group
    return [{"order": n, "id": i, "reason": why} for n, (i, why) in enumerate(rows)]


def read_pool(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return [{**r, "order": int(r["order"])} for r in csv.DictReader(f)]


def write_pool(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, POOL_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _matching_lines(lines: list[str], displayed: str | None) -> list[int]:
    if not displayed:
        return []
    needle = " ".join(displayed.split())
    pat = re.compile(rf"(?<![\d,.]){re.escape(needle)}(?![\d,]|\.\d)")
    return [i for i, line in enumerate(lines) if pat.search(line)]


def parse_printed(text: str) -> Decimal:
    """ "1,234.5", "(50)", "$-0.25" -> Decimal; raises ValueError."""
    s = str(text).replace(",", "").replace("$", "").replace(" ", "").strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        raise ValueError(f"not a number: {text!r}")
    try:
        d = Decimal(s)
    except InvalidOperation as e:
        raise ValueError(f"not a number: {text!r}") from e
    return -d if negative else d


def summary(pool: list[dict], checks: dict[tuple[str, str], dict]) -> dict:
    """Progress, and fixes per field in the random part of the pool (the answer key's error rate)."""
    out = {"filings": len(pool), "filings_done": 0, "verdicts": Counter(), "random_fields": 0, "random_changed": 0}
    for row in pool:
        verdicts = [checks.get((row["id"], n), {}).get("verdict") for n in FIELD_NAMES]
        out["filings_done"] += all(verdicts)
        out["verdicts"].update(v for v in verdicts if v)
        if row["reason"].startswith("random"):
            out["random_fields"] += sum(bool(v) for v in verdicts)
            out["random_changed"] += sum(v in ("fix", "drop") for v in verdicts)
    return out


class Review:
    def __init__(self, records: list[dict], pool: list[dict], handcheck_path: Path):
        self.records = {r["id"]: r for r in records}
        self.pool = [p for p in pool if p["id"] in self.records]
        self.path = Path(handcheck_path)
        self.lock = threading.Lock()
        self.checks = read_handcheck(self.path)

    def pool_view(self) -> list[dict]:
        out = []
        for p in self.pool:
            done = sum((p["id"], n) in self.checks for n in FIELD_NAMES)
            out.append({**p, "done": done, "fields": len(FIELD_NAMES)})
        return out

    def record_view(self, rid: str) -> dict:
        rec = self.records[rid]
        lines = rec["input"].split("\n")
        fields = []
        for name in FIELD_NAMES:
            gold = rec["gold"][name]
            fields.append(
                {
                    "name": name,
                    "gold": gold,
                    "flags": rec["flags"][name],
                    "lines": _matching_lines(lines, gold and gold["displayed"]),
                    "related": related_facts(rec, name),
                    "check": self.checks.get((rid, name)),
                }
            )
        reason = next((p["reason"] for p in self.pool if p["id"] == rid), "")
        keys = ("id", "ticker", "sector", "fiscal_year", "period_end", "filing_date", "accession", "record_flags")
        return {
            **{k: rec[k] for k in keys},
            "reason": reason,
            "edgar": edgar_url(rec),
            "lines": lines,
            "fields": fields,
        }

    def save(self, body: dict) -> dict:
        """Apply one verdict ({id, field, verdict, printed?, scale?, period_end?, note?}); verdict "" clears it."""
        rid, name, verdict = body.get("id"), body.get("field"), body.get("verdict")
        if rid not in self.records or not any(p["id"] == rid for p in self.pool):
            raise ValueError("unknown filing")
        if name not in FIELD_NAMES:
            raise ValueError("unknown field")
        if verdict not in (*VERDICTS, ""):
            raise ValueError("unknown verdict")
        note = str(body.get("note") or "")[:MAX_NOTE]
        gold = self.records[rid]["gold"][name]
        row = {"id": rid, "field": name, "verdict": verdict, "note": note}
        if verdict == "fix":
            scale = body.get("scale") or (gold and gold["scale"])
            if scale not in SCALES:
                raise ValueError("pick a scale")
            value = parse_printed(body.get("printed", "")) * (Decimal(10) ** SCALE_POWER[scale])
            period_end = body.get("period_end") or (gold and gold["period_end"]) or ""
            try:
                date.fromisoformat(period_end)
            except ValueError as e:
                raise ValueError("period end must be YYYY-MM-DD") from e
            row |= {"value": format(value.normalize(), "f"), "scale": scale, "period_end": period_end}
        row["checked_at"] = datetime.now().isoformat(timespec="seconds")
        with self.lock:
            if verdict:
                self.checks[(rid, name)] = row
            else:
                self.checks.pop((rid, name), None)
            write_handcheck(self.checks, self.path)
        return row


def _handler(review: Review, port: int):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _host_ok(self) -> bool:
            # Only this machine's own page: blocks DNS-rebinding pages from reading or writing
            if self.headers.get("Host") not in hosts:
                self._send(403, b"forbidden", "text/plain")
                return False
            return True

        def do_GET(self):
            if not self._host_ok():
                return
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/app.js":
                self._send(200, SCRIPT.encode(), "text/javascript; charset=utf-8")
            elif url.path == "/app.css":
                self._send(200, STYLE.encode(), "text/css; charset=utf-8")
            elif url.path == "/api/pool":
                self._json(200, {"pool": review.pool_view(), "summary": _jsonable(summary(review.pool, review.checks))})
            elif url.path == "/api/record":
                rid = parse_qs(url.query).get("id", [""])[0]
                if rid not in review.records:
                    self._json(404, {"error": "unknown filing"})
                else:
                    self._json(200, review.record_view(rid))
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._host_ok():
                return
            # A custom header and a JSON body: a cross-site form can't send either without a preflight
            if (
                urlparse(self.path).path != "/api/check"
                or self.headers.get("X-Handcheck") != "1"
                or not (self.headers.get("Content-Type") or "").startswith("application/json")
            ):
                self._json(403, {"error": "forbidden"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self._json(413, {"error": "too large"})
                return
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                self._json(200, {"saved": review.save(body)})
            except (ValueError, TypeError, AttributeError) as e:
                self._json(400, {"error": str(e)})

    return Handler


def _jsonable(s: dict) -> dict:
    return {**s, "verdicts": dict(s["verdicts"])}


def serve(review: Review, port: int = PORT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), BaseHTTPRequestHandler)
    server.RequestHandlerClass = _handler(review, server.server_address[1])  # the bound port (port=0 in tests)
    return server


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hand-check</title><link rel="stylesheet" href="/app.css"></head>
<body>
<aside><h1>Hand-check</h1><p id="progress"></p><ol id="pool"></ol></aside>
<main>
  <header id="head"><p>Pick a filing on the left.</p></header>
  <section id="fields"></section>
  <section><h2>Model input</h2><div id="input"></div></section>
</main>
<script src="/app.js"></script>
</body></html>
"""

STYLE = """
:root { --bg: #fff; --fg: #1d1d1f; --muted: #6e6e73; --line: #e2e2e6; --hl: #fff2b3; --hl2: #ffd75e;
  --ok: #1f7a3a; --fix: #a15c00; --drop: #b3261e; --panel: #f6f6f8; --link: #0b57d0; }
@media (prefers-color-scheme: dark) { :root { --bg: #18181b; --fg: #ececf0; --muted: #a1a1aa; --line: #34343a;
  --hl: #4a4220; --hl2: #7a6516; --ok: #5fd08a; --fix: #f2b25c; --drop: #ff8a80; --panel: #222227; --link: #8ab4f8; } }
* { box-sizing: border-box; }
body { margin: 0; display: flex; height: 100vh; font: 14px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
aside { width: 260px; overflow: auto; border-right: 1px solid var(--line); padding: 12px; background: var(--panel); }
aside h1 { font-size: 16px; margin: 0 0 4px; }
#progress { color: var(--muted); margin: 0 0 8px; }
#pool { list-style: none; padding: 0; margin: 0; }
#pool li { padding: 4px 6px; border-radius: 4px; cursor: pointer; display: flex; justify-content: space-between; gap: 6px; }
#pool li:hover { background: var(--line); }
#pool li.current { outline: 2px solid var(--fg); }
#pool li.done { color: var(--muted); }
#pool li small { color: var(--muted); }
main { flex: 1; overflow: auto; padding: 12px 20px; }
header h2 { margin: 0; font-size: 18px; }
header p { margin: 2px 0; color: var(--muted); }
.bar { display: flex; gap: 8px; margin: 8px 0; flex-wrap: wrap; }
button { font: inherit; padding: 4px 10px; border: 1px solid var(--line); border-radius: 4px; background: var(--bg); color: var(--fg); cursor: pointer; }
button:hover { border-color: var(--fg); }
button.primary { background: var(--fg); color: var(--bg); }
table { border-collapse: collapse; width: 100%; }
td, th { border-bottom: 1px solid var(--line); padding: 6px; vertical-align: top; text-align: left; }
th { font-weight: 600; color: var(--muted); }
tr.field { cursor: pointer; }
tr.field.sel { background: var(--panel); }
.flags { color: var(--fix); font-size: 12px; }
.related { color: var(--muted); font-size: 12px; }
.v-ok { color: var(--ok); font-weight: 600; } .v-fix { color: var(--fix); font-weight: 600; } .v-drop { color: var(--drop); font-weight: 600; }
form.fix { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
form.fix input, form.fix select { font: inherit; padding: 2px 4px; background: var(--bg); color: var(--fg); border: 1px solid var(--line); }
#input { font: 12px/1.4 ui-monospace, monospace; white-space: pre-wrap; word-break: break-word; }
#input div { padding: 0 4px; }
#input div.hl { background: var(--hl); }
#input div.hl.sel { background: var(--hl2); }
#input span.n { color: var(--muted); display: inline-block; width: 4em; user-select: none; }
.err { color: var(--drop); }
a { color: var(--link); }
td.date { white-space: nowrap; }
.snip { font: 12px/1.4 ui-monospace, monospace; background: var(--hl); padding: 2px 4px; margin-top: 4px; white-space: pre-wrap; word-break: break-word; }
"""

SCRIPT = r"""
"use strict";
const POWER = {units: 0, thousands: 3, millions: 6, billions: 9};
const SCALES = Object.keys(POWER);
let pool = [], current = null, rec = null, selField = null;

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids) if (k !== null && k !== undefined) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
const fmt = (x, d = 4) => Number(x).toLocaleString("en-US", {maximumFractionDigits: d});
function shown(gold) {  // the figure as the filing prints it
  const p = POWER[gold.scale] || 0;
  return fmt(gold.value / 10 ** p) + (gold.scale && gold.scale !== "units" ? " " + gold.scale : "");
}

async function api(path, body) {
  const opts = body ? {method: "POST", headers: {"Content-Type": "application/json", "X-Handcheck": "1"}, body: JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

async function loadPool() {
  const j = await api("/api/pool");
  pool = j.pool;
  const s = j.summary;
  const rate = s.random_fields ? ` · random sample: ${s.random_changed} of ${s.random_fields} fields changed` : "";
  document.getElementById("progress").textContent = `${s.filings_done} of ${s.filings} filings done${rate}`;
  const ol = document.getElementById("pool");
  ol.replaceChildren(...pool.map(p => el("li", {
      class: (p.done === p.fields ? "done" : "") + (p.id === current ? " current" : ""),
      title: p.reason, onclick: () => open(p.id)},
    el("span", {}, (p.done === p.fields ? "✓ " : "") + p.id),
    el("small", {}, p.reason.startsWith("random") ? "random" : p.reason.split(";").length + " flag(s)"))));
}

async function open(id) {
  current = id; selField = null;
  rec = await api("/api/record?id=" + encodeURIComponent(id));
  render();
  loadPool();
  document.querySelector("main").scrollTop = 0;
}

function render() {
  const h = document.getElementById("head");
  h.replaceChildren(
    el("h2", {}, `${rec.id} · ${rec.ticker} · fiscal year ${rec.fiscal_year}`),
    el("p", {}, `${rec.sector} · period end ${rec.period_end} · filed ${rec.filing_date} · why: ${rec.reason || "-"}`),
    el("div", {class: "bar"},
      el("button", {class: "primary", onclick: allOk, title: "Shortcut: a"}, "All correct → next (a)"),
      el("button", {onclick: () => step(1), title: "Shortcut: j"}, "Next (j)"),
      el("button", {onclick: () => step(-1), title: "Shortcut: k"}, "Previous (k)"),
      el("a", {href: rec.edgar, target: "_blank", rel: "noopener noreferrer"}, "Open on EDGAR ↗")));
  const rows = rec.fields.map(f => fieldRow(f));
  document.getElementById("fields").replaceChildren(el("table", {},
    el("thead", {}, el("tr", {}, ...["Field", "Gold (as printed)", "Period end", "Tag · flags", "Other tagged values, same period", "Verdict"].map(t => el("th", {}, t)))),
    el("tbody", {}, ...rows)));
  renderInput();
}

function fieldRow(f) {
  const g = f.gold;
  const verdict = f.check ? el("span", {class: "v-" + f.check.verdict}, f.check.verdict +
    (f.check.verdict === "fix" ? ` → ${fmt(f.check.value)} (${f.check.scale}, ${f.check.period_end})` : "")) : el("span", {}, "-");
  const cell = el("td", {}, verdict, el("div", {class: "bar"},
    el("button", {onclick: e => { e.stopPropagation(); save(f.name, "ok"); }}, "OK"),
    el("button", {onclick: e => { e.stopPropagation(); showFix(f, cell); }}, g ? "Fix" : "Add"),
    g ? el("button", {onclick: e => { e.stopPropagation(); save(f.name, "drop"); }}, "Drop") : null,
    f.check ? el("button", {onclick: e => { e.stopPropagation(); save(f.name, ""); }}, "Clear") : null));
  const related = f.related.filter(r => !g || r.tag !== g.tag || r.value !== g.value).map(r =>
    el("div", {}, `${r.tag.split(":")[1]}: ${fmt(r.value)}${r.start ? "" : " @ " + r.end}`));
  const flags = f.flags.filter(x => x !== "missing" || !g);
  const tr = el("tr", {class: "field" + (selField === f.name ? " sel" : ""), onclick: () => select(f.name)},
    el("td", {}, el("b", {}, f.name), f.lines.length ? null : el("div", {class: "related"}, g ? "not found in input" : "")),
    el("td", {}, g ? el("div", {}, shown(g), el("div", {class: "related"}, `printed "${g.displayed}" = ${fmt(g.value)}`),
        ...f.lines.slice(0, 2).map(i => el("div", {class: "snip", title: "input line " + (i + 1)}, rec.lines[i]))) : el("i", {}, "unscored")),
    el("td", {class: "date"}, g ? g.period_end : ""),
    el("td", {}, g ? g.tag.split(":")[1] : "", el("div", {class: "flags"}, flags.join(", "))),
    el("td", {class: "related"}, ...related),
    cell);
  return tr;
}

function showFix(f, cell) {
  const g = f.gold || {};
  const printed = el("input", {placeholder: "as printed, e.g. (1,234.5)", size: "18", value: g.displayed || ""});
  const scale = el("select", {}, ...SCALES.map(s => { const o = el("option", {value: s}, s); if (s === (g.scale || (f.name === "eps_diluted" ? "units" : "millions"))) o.selected = true; return o; }));
  const period = el("input", {placeholder: "YYYY-MM-DD", size: "11", value: g.period_end || (f.name === "shares_outstanding" ? "" : rec.period_end)});
  const note = el("input", {placeholder: "why (optional)", size: "24"});
  const msg = el("span", {class: "err"});
  const form = el("form", {class: "fix", onclick: e => e.stopPropagation(), onsubmit: async e => {
      e.preventDefault();
      try { await save(f.name, "fix", {printed: printed.value, scale: scale.value, period_end: period.value, note: note.value}); }
      catch (err) { msg.textContent = err.message; }
    }}, printed, scale, period, note, el("button", {type: "submit", class: "primary"}, "Save"), msg);
  cell.append(form);
  printed.focus();
}

async function save(field, verdict, extra) {
  await api("/api/check", {id: rec.id, field, verdict, ...(extra || {})});
  const keep = selField;
  rec = await api("/api/record?id=" + encodeURIComponent(rec.id));
  selField = keep;
  render();
  loadPool();
}

async function allOk() {
  for (const f of rec.fields) if (!f.check) await api("/api/check", {id: rec.id, field: f.name, verdict: "ok"});
  await loadPool();
  const i = pool.findIndex(p => p.id === rec.id);
  const next = pool.slice(i + 1).concat(pool.slice(0, i)).find(p => p.done < p.fields);
  if (next) open(next.id); else open(rec.id);
}

function step(d) {
  const i = pool.findIndex(p => p.id === current);
  const j = (i + d + pool.length) % pool.length;
  if (pool.length) open(pool[j].id);
}

function select(name) {
  selField = name;
  render();
  const f = rec.fields.find(x => x.name === name);
  if (f && f.lines.length) document.getElementById("L" + f.lines[0]).scrollIntoView({block: "center"});
}

function renderInput() {
  const hl = new Map();
  for (const f of rec.fields) for (const i of f.lines) hl.set(i, (hl.get(i) || []).concat(f.name));
  document.getElementById("input").replaceChildren(...rec.lines.map((line, i) => {
    const names = hl.get(i) || [];
    const cls = names.length ? "hl" + (names.includes(selField) ? " sel" : "") : "";
    return el("div", {id: "L" + i, class: cls, title: names.join(", ")}, el("span", {class: "n"}, i + 1), line);
  }));
}

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || !rec) return;
  if (e.key === "a") allOk(); else if (e.key === "j") step(1); else if (e.key === "k") step(-1);
});

loadPool().then(() => { const first = pool.find(p => p.done < p.fields) || pool[0]; if (first) open(first.id); });
"""
