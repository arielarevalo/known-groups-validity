"""Where the logit monitor reads (`design 0028`): the start of the reasoning trace
and its sentence boundaries. The published method scores every boundary; each
position is one teacher-forced forward over the whole prefix here, so the profile
caps the count (`logit_max_positions`) and the boundaries are spread evenly over
the trace. The cap and the chosen offsets are written with every score."""

from __future__ import annotations

from cv.whitebox import constants


def find_boundaries(text: str) -> list[int]:
    """Character offsets at which a new sentence starts, excluding offset 0 and the
    end of the text."""
    offsets = []
    for match in constants.sentence_rules.boundary.finditer(text):
        end = match.end()
        if 0 < end < len(text):
            offsets.append(end)
    return sorted(set(offsets))


def choose_positions(text: str, max_positions: int) -> list[int]:
    """The offsets to score: 0 (no reasoning yet) plus up to ``max_positions - 1``
    boundaries spread evenly over the trace, in order."""
    boundaries = find_boundaries(text)
    n_boundaries = max(0, int(max_positions) - 1)
    if n_boundaries == 0 or not boundaries:
        return [0]
    if len(boundaries) <= n_boundaries:
        return [0, *boundaries]
    if n_boundaries == 1:
        return [0, boundaries[-1]]
    last = len(boundaries) - 1
    chosen = [boundaries[round(step * last / (n_boundaries - 1))] for step in range(n_boundaries)]
    return [0, *sorted(set(chosen))]


def attach_sentence(sentence: str, prefix: str) -> str:
    """The probe sentence as it is appended: separated from a non-empty prefix by
    a space unless the prefix already ends in whitespace."""
    if prefix and not prefix[-1].isspace():
        return " " + sentence
    return sentence
