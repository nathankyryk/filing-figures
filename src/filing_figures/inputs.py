"""Build the model's input from a 10-K: the cover page plus the three main financial statements.

The statements are found the way a person would find them, never through the XBRL tags:
  1. A title: a short block (text, or a small heading table) right above the table, or in its
     first rows, that reads like the statement ("Consolidated Statements of Operations",
     "Balance Sheets", "Statements of Cash Flows") and not like a reference to it ("Analysis of
     the Consolidated Statements of Income", "Condensed ... (Parent Company)").
  2. A key line with numbers: revenue or sales (income statement), total assets (balance sheet),
     operating activities (cash flows). A statement of comprehensive income alone has no revenue
     line; an index or table of contents has no numbers.
  3. The first such table after the "Item 8" heading wins. MD&A summary tables come earlier.
  4. No titled table? Then the first table after Item 8 with all of the statement's key lines
     (revenue + net income + diluted EPS; total assets + total liabilities; operating + investing +
     financing activities): a few filers print the statement with no title at all.

The tags are used afterwards only to check the selection (see `coverage`), and the hidden
`ix:header` block, which repeats tagged facts, is removed before any text is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from filing_figures import ixbrl

I = re.IGNORECASE
TITLES = {
    "income_statement": re.compile(
        r"statements?\s+of\s+(consolidated\s+)?(comprehensive\s+)?(operations|income|earnings|loss)"
        r"|income\s+statements?|consolidated\s+results\s+of\s+operations",
        I,
    ),
    "balance_sheet": re.compile(
        r"balance\s+sheets?|statements?\s+of\s+(consolidated\s+)?financial\s+(position|condition)", I
    ),
    "cash_flow_statement": re.compile(r"statements?\s+of\s+(consolidated\s+)?cash\s+flows?", I),
}
REVENUE_LINE = re.compile(r"\b(revenues?|sales)\b", I)
NET_INCOME_LINE = re.compile(r"net\s+(income|earnings|loss)", I)
# Each statement's key lines: a titled table needs one of them (a row label with 2+ numbers in the row:
# statements show at least two years, an index shows one page number). The income statement's
# revenue line may have no label (a bare total under "Revenue:"); then net income counts, unless the
# title says "comprehensive": that statement starts from net income and has no revenue at all.
# Many balance sheets leave the total-assets line unlabelled too.
# McDonald's and Disney say "Cash provided by operations" rather than "operating activities"
OPERATING_LINE = re.compile(
    r"operating\s+activities|(provided|generated|used)\s+(by|in|from)\s+(continuing\s+)?operations"
    r"|cash\s+flows?\s+from\s+operations",
    I,
)
KEY_LINES = {
    "income_statement": (REVENUE_LINE, NET_INCOME_LINE),
    "balance_sheet": (re.compile(r"total\s+(current\s+)?assets|total\s+liabilities", I),),
    "cash_flow_statement": (OPERATING_LINE,),
}
# With no title, a table needs all of these to pass for the statement
ALL_KEY_LINES = {
    "income_statement": (REVENUE_LINE, NET_INCOME_LINE, re.compile(r"diluted", I)),
    "balance_sheet": (re.compile(r"total\s+(current\s+)?assets", I), re.compile(r"total\s+liabilities", I)),
    "cash_flow_statement": (
        OPERATING_LINE,
        re.compile(r"investing\s+activities", I),
        re.compile(r"financing\s+activities", I),
    ),
}
STATEMENTS_WITHIN = 250  # blocks from the "Item 8" (or auditor's report) heading to the last statement
# Words that make a title-like line a reference to a statement, or a statement we don't want
NOT_A_TITLE = re.compile(
    r"\b(analysis|summary|summarized|following|discussion|see|refer|selected|condensed|supplemental|"
    r"consolidating|parent|registrant|unaudited|quarterly|pro\s+forma|notes?\s+to|index)\b",
    I,
)
AUDIT_REPORT = re.compile(r"^\W*report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm", I)
TITLE_MAX_CHARS, TITLE_MAX_POS = 200, 80
SMALL_TABLE = 300  # a table this short, with 3 rows or fewer, is a heading or a page footer, not data
ITEM8 = re.compile(r"^\W*item\s*8\b", I)
# The cover ends at the first "PART I" heading or at the table of contents (a table listing Item 1A
# and Item 7). Not at a "Table of Contents" line: many filings repeat that link atop every page,
# including the cover's second page, where the shares-outstanding sentence sits.
COVER_END = re.compile(r"^\W*part\s+i\b", I)
TOC_TABLE = (re.compile(r"item\s*1a|risk\s+factors", I), re.compile(r"item\s*7\b", I))
NUMBER = re.compile(r"\d[\d,]*(\.\d+)?")
INVISIBLE = dict.fromkeys((0x200B, 0x200C, 0x200D, 0xFEFF), None)  # zero-width characters
BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "center", "font", "td"}
TITLE_BLOCKS, TITLE_CHARS = 6, 800
COVER_CHARS = 8000

# The section each field is read from, for the coverage check
FIELD_SECTION = {
    "revenue": "income_statement",
    "net_income": "income_statement",
    "eps_diluted": "income_statement",
    "total_assets": "balance_sheet",
    "operating_cash_flow": "cash_flow_statement",
    "shares_outstanding": "cover",
}


def _local(el) -> str:
    return etree.QName(el).localname.lower() if isinstance(el.tag, str) else ""


def _norm(s: str) -> str:
    return " ".join(s.translate(INVISIBLE).split())


def _text(el) -> str:
    return _norm("".join(el.itertext()))


def _hidden(el) -> bool:
    return "display:none" in (el.get("style") or "").replace(" ", "").lower()


def _drop(el: etree._Element) -> None:
    """Remove an element but keep the text that follows it (lxml stores that as its `tail`)."""
    parent = el.getparent()
    if parent is None:
        return
    if el.tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def strip_hidden(root: etree._Element) -> None:
    """Drop the ix:header (hidden facts and contexts) and anything styled display:none."""
    for el in list(root.iter()):
        if isinstance(el.tag, str) and (el.tag == f"{{{ixbrl.IX}}}header" or _hidden(el)):
            _drop(el)


@dataclass
class Block:
    kind: str  # "text" or "table"
    el: etree._Element
    text: str

    @property
    def small(self) -> bool:
        """Text, or a table short enough to be a heading or footer."""
        return self.kind == "text" or (len(self.text) < SMALL_TABLE and len(table_rows(self.el)) <= 3)


def _inline_text(el: etree._Element) -> str:
    """An element's own text, leaving out child blocks and tables (their tails stay)."""
    parts = [el.text or ""]
    for c in el:
        if _local(c) not in BLOCK_TAGS | {"table"}:
            parts.append("".join(c.itertext()))
        parts.append(c.tail or "")
    return _norm("".join(parts))


def blocks(root: etree._Element) -> list[Block]:
    """The body as a flat list of text blocks and (outermost) tables, in document order."""
    out: list[Block] = []

    def walk(el):
        for child in el:
            name = _local(child)
            if not name:
                continue
            if name == "table":
                out.append(Block("table", child, _text(child)))
            elif name in BLOCK_TAGS and not any(_local(d) in BLOCK_TAGS | {"table"} for d in child.iterdescendants()):
                t = _text(child)
                if t:
                    out.append(Block("text", child, t))
            else:
                # A block holding both text and child blocks: keep its own text (a title can sit there)
                own = _inline_text(child) if name in BLOCK_TAGS else ""
                if own:
                    out.append(Block("text", child, own))
                walk(child)

    body = next((el for el in root.iter() if _local(el) == "body"), root)
    walk(body)
    return out


def table_rows(table: etree._Element) -> list[list[str]]:
    """Each row's non-empty cells, with "$", "(" and ")" / "%" joined to their number."""
    rows = []
    for tr in table.iter():
        if _local(tr) != "tr":
            continue
        cells = [_text(td) for td in tr if _local(td) in {"td", "th"}]
        merged: list[str] = []
        carry = ""
        for c in cells:
            if not c:
                continue
            if c in {"$", "(", "$(", "($"}:
                carry += c
                continue
            if merged and re.fullmatch(r"\)?%?\)?", c):
                merged[-1] += c
                continue
            merged.append((carry + " " + c).strip() if carry == "$" else carry + c)
            carry = ""
        if merged:
            rows.append(merged)
    return rows


def table_text(table: etree._Element) -> str:
    return "\n".join(" | ".join(r) for r in table_rows(table))


def _row_labels(table: etree._Element) -> list[str]:
    """Labels of the rows with two or more numbers (two years of figures; an index row has one page number)."""
    return [r[0] for r in table_rows(table) if sum(bool(NUMBER.search(c)) for c in r[1:]) >= 2]


def _key_lines(table: etree._Element, patterns, labels: list[str] | None = None) -> bool:
    """Every pattern names some row that has figures in it."""
    labels = _row_labels(table) if labels is None else labels
    return all(any(p.search(lab) for lab in labels) for p in patterns)


def is_title(text: str, title_re: re.Pattern) -> bool:
    if len(text) > TITLE_MAX_CHARS or NOT_A_TITLE.search(text):
        return False
    m = title_re.search(text)
    return m is not None and m.start() <= TITLE_MAX_POS


def _above(bs: list[Block], i: int) -> list[int]:
    """The small blocks just above table i: where a title and the scale note sit."""
    out, size = [], 0
    for j in range(i - 1, max(-1, i - 1 - TITLE_BLOCKS), -1):
        if not bs[j].small or size > TITLE_CHARS:
            break
        out.insert(0, j)
        size += len(bs[j].text)
    return out


SCALE_NOTE = re.compile(r"thousands|millions|billions|except\s+(per[\s-]share|share)", I)


def _titled(bs: list[Block], i: int, title_re: re.Pattern) -> tuple[str | None, list[int]]:
    """Table i's title, if it is titled like the statement, and the caption to keep: from the title
    down to the table, or, when the title is inside the table, the scale note above it (if any)."""
    above = _above(bs, i)
    start = next((k for k, j in enumerate(above) if is_title(bs[j].text, title_re)), None)
    if start is not None:
        return bs[above[start]].text, above[start:]
    in_rows = next((" ".join(r) for r in table_rows(bs[i].el)[:3] if is_title(" ".join(r), title_re)), None)
    return in_rows, [j for j in above if SCALE_NOTE.search(bs[j].text)]


def _has_key_line(name: str, labels: list[str], title: str) -> bool:
    if name == "income_statement":
        if any(REVENUE_LINE.search(lab) for lab in labels):
            return True
        return "comprehensive" not in title.lower() and any(NET_INCOME_LINE.search(lab) for lab in labels)
    return any(p.search(lab) for p in KEY_LINES[name] for lab in labels)


@dataclass
class Selection:
    cover: list[Block] = field(default_factory=list)
    statements: dict[str, tuple[list[Block], Block]] = field(default_factory=dict)  # name -> (caption, table)
    how: dict[str, str] = field(default_factory=dict)  # name -> "title" | "structure"


def _cover(bs: list[Block]) -> list[Block]:
    out, size = [], 0
    for b in bs:
        if b.kind == "text" and COVER_END.search(b.text):
            break
        if b.kind == "table" and all(p.search(b.text) for p in TOC_TABLE):
            break
        if size > COVER_CHARS:
            break
        out.append(b)
        size += len(b.text)
    return out


def select(bs: list[Block]) -> Selection:
    sel = Selection(cover=_cover(bs))
    titled: dict[str, list[tuple[int, list[int]]]] = {name: [] for name in TITLES}
    for i, b in enumerate(bs):
        if b.small:
            continue
        labels = None
        for name, title_re in TITLES.items():
            title, caption = _titled(bs, i, title_re)
            if title is None:
                continue
            labels = _row_labels(b.el) if labels is None else labels
            if _has_key_line(name, labels, title):
                titled[name].append((i, caption))
    # Where the statements start: the first "Item 8" heading or auditor's report heading (Intel's
    # 10-K has no Item headings) that every statement found follows closely. The table of contents
    # sits hundreds of blocks before any statement; MD&A tables that look like statements come before
    # Item 8; and combined filings (a utility and its subsidiaries) and parent-only schedules repeat
    # the headings and statements later on.
    marks = [i for i, b in enumerate(bs) if b.small and (ITEM8.search(b.text) or AUDIT_REPORT.search(b.text))]
    found = [c for c in titled.values() if c]
    close = [p for p in marks if all(any(p < t <= p + STATEMENTS_WITHIN for t, _ in c) for c in found)]
    after_all = [p for p in marks if all(any(t > p for t, _ in c) for c in found)]
    anchor = close[0] if close else max(after_all, default=-1)
    for name, cands in titled.items():
        after = [c for c in cands if c[0] > anchor]
        if after or cands:
            i, caption = (after or cands)[0]
            sel.statements[name], sel.how[name] = ([bs[j] for j in caption], bs[i]), "title"
            continue
        # No titled table: the first one after Item 8 with every key line of the statement
        for i, b in enumerate(bs):
            if i > anchor and not b.small and _key_lines(b.el, ALL_KEY_LINES[name]):
                caption = [bs[j] for j in _above(bs, i) if SCALE_NOTE.search(bs[j].text)]
                sel.statements[name], sel.how[name] = (caption, b), "structure"
                break
    return sel


def _block_text(b: Block) -> str:
    return table_text(b.el) if b.kind == "table" else b.text


def render(sel: Selection) -> str:
    parts = ["## Cover page", "\n".join(_block_text(b) for b in sel.cover)]
    for name in TITLES:
        if name not in sel.statements:
            continue
        caption, table = sel.statements[name]
        parts += [
            f"## {name.replace('_', ' ').capitalize()}",
            "\n".join(_block_text(b) for b in caption),
            table_text(table.el),
        ]
    return "\n\n".join(p for p in parts if p)


@dataclass
class Built:
    text: str  # the model input
    selection: Selection
    root: etree._Element  # the tree, with the hidden header removed
    contexts: dict  # read before the removal: the hidden header holds them
    dei: dict  # the cover's document facts, also often hidden


def build(html: bytes) -> Built:
    """The model input for one 10-K, and what is needed to check it."""
    root = ixbrl.parse(html)
    ctx, doc = ixbrl.contexts(root), ixbrl.dei(root)
    strip_hidden(root)
    sel = select(blocks(root))
    return Built(render(sel), sel, root, ctx, doc)


def _section_elements(sel: Selection, section: str) -> list[etree._Element]:
    if section == "cover":
        return [b.el for b in sel.cover]
    if section not in sel.statements:
        return []
    caption, table = sel.statements[section]
    return [table.el, *(b.el for b in caption)]


def coverage(sel: Selection, root: etree._Element, ctx: dict, gold: dict) -> dict:
    """For each gold field: is its tagged number inside the selected section, and with which scale?

    gold: {field: {"value", "start", "end", "tag"} | None} from labels.label_filing.
    Returns {field: {"in_input": bool, "scale": str|None, "text": str|None, "mismatch": bool}}.
    """
    out = {}
    for name, g in gold.items():
        if g is None:
            out[name] = None
            continue
        found = None
        for el in _section_elements(sel, FIELD_SECTION[name]):
            for f in ixbrl.facts(root, within=el, ctx=ctx):
                if f.name != g["tag"] or f.context.has_dims or f.context.end != g["end"]:
                    continue
                if (f.context.start or None) != (g["start"] or None):
                    continue
                found = f
                break
            if found:
                break
        if found is None:
            out[name] = {"in_input": False, "scale": None, "text": None, "mismatch": False}
            continue
        tol = max(abs(g["value"]) * 1e-6, 0.5 * 10**found.scale if found.scale > 0 else 0.005)
        mismatch = found.value is None or abs(found.value - g["value"]) > tol
        out[name] = {"in_input": True, "scale": ixbrl.scale_name(found.scale), "text": found.text, "mismatch": mismatch}
    return out
