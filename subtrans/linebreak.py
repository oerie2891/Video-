"""Subtitle line wrapping.

Naive `textwrap` output reads badly on screen: it leaves orphaned function
words at the end of a line ("in the / kitchen") and wildly unbalanced lines.
This module picks break points by minimising line-length variance plus a
linguistic penalty, which is what professional subtitling guidelines ask for.

Non-space scripts (Chinese, Japanese, Thai, Khmer...) break on characters with
kinsoku-style rules: a line never starts with closing punctuation and never
ends with opening punctuation.
"""

from __future__ import annotations

from .languages import CueStyle

# Words that should not be left dangling at the end of a line. Covering the
# major Latin-script languages; other languages fall back to punctuation and
# balance alone, which is still better than greedy wrapping.
_DANGLING: dict[str, frozenset[str]] = {
    "en": frozenset("""a an the and or but nor so yet for of to in on at by with from
        into onto upon over under as if then than that this these those my your his her
        its our their is are was were be been being am do does did has have had will
        would can could shall should may might must not no very more most such about
        after before between during through without within because while when where who
        whom whose which what how""".split()),
    "es": frozenset("""el la los las un una unos unas y o pero ni de del al a en con por
        para sin sobre entre desde hasta que como cuando donde quien cual su sus mi mis
        tu tus es son era eran ser estar muy mas más no ya si""".split()),
    "fr": frozenset("""le la les un une des du de au aux et ou mais ni à en dans sur
        sous par pour sans avec chez vers que qui dont où quand comme si son sa ses mon
        ma mes ton ta tes est sont était très plus ne pas y il elle on nous vous""".split()),
    "de": frozenset("""der die das den dem des ein eine einen einem einer und oder aber
        noch von zu in an auf bei mit nach über unter für ohne durch gegen um als wie
        wenn dass weil sein ihre seine mein dein ist sind war waren sehr nicht kein
        schon auch""".split()),
    "it": frozenset("""il lo la i gli le un uno una dei del della e o ma né di a da in
        su con per tra fra che chi cui come quando dove se suo sua mio mia è sono era
        erano molto più non già anche""".split()),
    "pt": frozenset("""o a os as um uma uns umas e ou mas nem de do da dos das em no na
        nos nas por para sem sobre com entre que quem como quando onde seu sua meu minha
        é são era eram muito mais não já também""".split()),
    "nl": frozenset("""de het een en of maar noch van te in op aan bij met naar over
        onder voor zonder door tegen om als dat die dit deze wie welke hoe wanneer waar
        zijn is was waren heel zeer meer niet geen al ook""".split()),
}
_DANGLING["ca"] = _DANGLING["es"]
_DANGLING["gl"] = _DANGLING["pt"]

_CLAUSE_END = tuple(",;:—–")
_SENTENCE_END = tuple(".!?…。！？；")
_CJK_NO_LINE_START = "。，、！？：；）〕］｝」』”’・ー々ヽヾ…‥ﾞﾟ!?,.:;)]}»"
_CJK_NO_LINE_END = "（〔［｛「『“‘([{«"


def _dangling_set(lang: str | None) -> frozenset[str]:
    if not lang:
        return _DANGLING["en"]
    return _DANGLING.get(lang.lower(), frozenset())


def wrap(text: str, style: CueStyle, lang: str | None = None) -> list[str]:
    """Wrap `text` into at most `style.max_lines` balanced lines.

    If the text genuinely cannot fit the budget, the best available split is
    returned anyway (over-long) — the caller decides whether to re-split the
    cue instead, which is the correct fix.
    """
    text = " ".join(text.split()) if style.breaks_on_spaces else text.strip()
    if not text:
        return []
    limit = style.max_chars_per_line
    if len(text) <= limit:
        return [text]

    if style.breaks_on_spaces:
        return _wrap_spaced(text, limit, style.max_lines, lang)
    return _wrap_unspaced(text, limit, style.max_lines)


def _wrap_spaced(text: str, limit: int, max_lines: int, lang: str | None) -> list[str]:
    tokens = text.split(" ")
    if len(tokens) == 1:
        return [text]

    # Prefix lengths so any line's rendered width is O(1) to compute.
    # width(i, j) = chars of tokens[i:j] joined by single spaces.
    cum = [0]
    for tok in tokens:
        cum.append(cum[-1] + len(tok))

    def width(i: int, j: int) -> int:
        return cum[j] - cum[i] + (j - i - 1)

    dangling = _dangling_set(lang)

    def break_penalty(i: int, j: int) -> float:
        """Cost of ending a line with tokens[i:j] and starting the next at j."""
        if j >= len(tokens):
            return 0.0
        last = tokens[j - 1]
        nxt = tokens[j]
        stripped = last.rstrip('"\'»”’)')
        penalty = 0.0
        if stripped.endswith(_SENTENCE_END):
            penalty -= 3.0          # strongest natural break
        elif stripped.endswith(_CLAUSE_END):
            penalty -= 2.0
        word = last.strip('.,;:!?"\'()[]«»“”‘’…').lower()
        if word in dangling:
            penalty += 3.0          # don't orphan "the", "de", "der"...
        if last.endswith("-"):
            penalty += 4.0          # never break a hyphenated compound
        if nxt and nxt[0] in ",.;:!?…)]}":
            penalty += 4.0
        if last in ("(", "[", "«", "“", "-", "–", "—"):
            penalty += 3.0
        if nxt.strip('-–—') == "" and len(nxt) <= 1:
            penalty += 2.0          # lone dash starting a line
        return penalty

    n = len(tokens)
    scale = (limit * limit) / 90.0  # makes one penalty point ≈ a few chars of imbalance

    for lines in range(2, max_lines + 1):
        # dp[k][j] = best cost using k lines to cover tokens[:j]
        dp: list[list[float]] = [[float("inf")] * (n + 1) for _ in range(lines + 1)]
        back: list[list[int]] = [[-1] * (n + 1) for _ in range(lines + 1)]
        dp[0][0] = 0.0
        for k in range(1, lines + 1):
            for j in range(1, n + 1):
                for i in range(k - 1, j):
                    if dp[k - 1][i] == float("inf"):
                        continue
                    w = width(i, j)
                    if w > limit:
                        continue  # hard constraint; relaxed below if unsatisfiable
                    cost = dp[k - 1][i] + w * w + break_penalty(i, j) * scale
                    if cost < dp[k][j]:
                        dp[k][j] = cost
                        back[k][j] = i
        if dp[lines][n] < float("inf"):
            # Reconstruct.
            cuts: list[int] = []
            j, k = n, lines
            while k > 0:
                i = back[k][j]
                cuts.append(i)
                j, k = i, k - 1
            cuts.reverse()
            bounds = cuts + [n]
            out = [" ".join(tokens[bounds[idx]:bounds[idx + 1]]) for idx in range(lines)]
            # Prefer the fewest lines that fit; stop at the first feasible count.
            return out

    # Nothing fits the per-line limit: fall back to greedy so we still emit
    # something readable, and let the cue splitter shorten the text upstream.
    return _greedy(tokens, limit, max_lines)


def _greedy(tokens: list[str], limit: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for tok in tokens:
        candidate = f"{cur} {tok}".strip()
        if cur and len(candidate) > limit and len(lines) < max_lines - 1:
            lines.append(cur)
            cur = tok
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines[:max_lines] if lines else [" ".join(tokens)]


def _wrap_unspaced(text: str, limit: int, max_lines: int) -> list[str]:
    """Character-based wrapping with kinsoku rules for CJK/Thai/Khmer."""
    chars = list(text)
    n = len(chars)
    target_lines = min(max_lines, max(2, -(-n // limit)))

    def legal(pos: int) -> bool:
        if pos <= 0 or pos >= n:
            return False
        if chars[pos] in _CJK_NO_LINE_START:
            return False
        if chars[pos - 1] in _CJK_NO_LINE_END:
            return False
        return True

    # Aim for even lines, then walk outward to the nearest legal break.
    cuts: list[int] = []
    for k in range(1, target_lines):
        ideal = round(n * k / target_lines)
        chosen = None
        for delta in range(0, limit):
            for cand in (ideal - delta, ideal + delta):
                if legal(cand) and (not cuts or cand > cuts[-1]):
                    chosen = cand
                    break
            if chosen is not None:
                break
        cuts.append(chosen if chosen is not None else ideal)

    bounds = [0] + cuts + [n]
    lines = ["".join(chars[bounds[i]:bounds[i + 1]]).strip() for i in range(len(bounds) - 1)]
    return [ln for ln in lines if ln]


def visual_len(text: str) -> int:
    """Length as a reader perceives it: newlines don't count toward CPS."""
    return len(text.replace("\n", " "))
