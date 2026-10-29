"""Read a 10-K's inline XBRL: the tagged numbers (with their scale and sign) and the cover's dei facts.

Every number in an inline-XBRL 10-K is wrapped in an `ix:nonFraction` tag that names its concept,
its context (period, and any dimensions such as a segment) and how it is displayed:
  <ix:nonFraction name="us-gaap:Revenues" contextRef="c-1" scale="6" decimals="-6" sign="-"
                  format="ixt:num-dot-decimal">391,035</ix:nonFraction>
`scale="6"` means the filing shows it "in millions": exactly the scale the model is asked to report,
which companyfacts does not carry. The value is displayed number x 10^scale, negated if sign="-".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lxml import etree

IX = "http://www.xbrl.org/2013/inlineXBRL"
XBRLI = "http://www.xbrl.org/2003/instance"
NS = {"ix": IX, "xbrli": XBRLI}
SCALE_NAMES = {0: "units", 3: "thousands", 6: "millions", 9: "billions"}
_WORDS = {"none": 0, "no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def parse(html: bytes) -> etree._Element:
    """Parse the XHTML safely: downloaded files are untrusted, so no entities and no network."""
    parser = etree.XMLParser(huge_tree=True, recover=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(html, parser)
    if root is None:
        raise ValueError("not parseable as XHTML")
    return root


@dataclass(frozen=True)
class Context:
    start: str | None
    end: str  # the instant, for instant contexts
    has_dims: bool


@dataclass
class Fact:
    name: str  # "us-gaap:Revenues"
    context: Context
    scale: int
    sign: int
    text: str  # as displayed
    value: float | None  # displayed number x 10^scale x sign
    element: etree._Element


def contexts(root: etree._Element) -> dict[str, Context]:
    out = {}
    for c in root.iter(f"{{{XBRLI}}}context"):
        period = c.find("xbrli:period", NS)
        if period is None:
            continue
        instant = period.findtext("xbrli:instant", namespaces=NS)
        start = period.findtext("xbrli:startDate", namespaces=NS)
        end = instant or period.findtext("xbrli:endDate", namespaces=NS)
        dims = c.find("xbrli:entity/xbrli:segment", NS) is not None or c.find("xbrli:scenario", NS) is not None
        if end:
            out[c.get("id")] = Context(None if instant else start.strip(), end.strip(), dims)
    return out


def displayed_number(text: str, fmt: str | None) -> float | None:
    """The number as shown, per the iXBRL transformation named in `format` (sign handled separately)."""
    t = text.strip().lower()
    fmt = (fmt or "").lower()
    if "zerodash" in fmt or "fixed-zero" in fmt or "fixedzero" in fmt or t in {"-", "—", "–"}:
        return 0.0
    if "numwordsen" in fmt or t in _WORDS:
        return float(_WORDS[t]) if t in _WORDS else None
    if "comma-decimal" in fmt or "numcommadecimal" in fmt:
        t = t.replace(".", "").replace(" ", "").replace(",", ".")
    digits = re.sub(r"[^0-9.]", "", t)
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def facts(root: etree._Element, within: etree._Element | None = None, ctx: dict | None = None) -> list[Fact]:
    """The tagged numbers in the document, or only those inside `within`."""
    ctx = ctx if ctx is not None else contexts(root)
    out = []
    for el in (within if within is not None else root).iter(f"{{{IX}}}nonFraction"):
        c = ctx.get(el.get("contextRef", ""))
        if c is None:
            continue
        text = "".join(el.itertext())
        scale = int(el.get("scale") or 0)
        sign = -1 if el.get("sign") == "-" else 1
        number = displayed_number(text, el.get("format"))
        value = None if number is None else sign * number * 10**scale
        out.append(Fact(el.get("name", ""), c, scale, sign, text.strip(), value, el))
    return out


def dei(root: etree._Element) -> dict[str, str]:
    """The cover's document facts: fiscal year focus, period end date, document type, amendment flag."""
    want = {
        "dei:DocumentFiscalYearFocus",
        "dei:DocumentPeriodEndDate",
        "dei:DocumentType",
        "dei:AmendmentFlag",
        "dei:EntityRegistrantName",
    }
    out = {}
    for el in root.iter(f"{{{IX}}}nonNumeric"):
        name = el.get("name")
        if name in want and name.split(":")[1] not in out:  # the first occurrence (the cover's)
            out[name.split(":")[1]] = " ".join("".join(el.itertext()).split())
    return out


def scale_name(scale: int) -> str:
    return SCALE_NAMES.get(scale, f"1e{scale}")
