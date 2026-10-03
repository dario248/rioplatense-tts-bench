"""WER and CER over the stored transcripts (SPEC 7.1).

Core and dependency-free: scoring costs milliseconds and gets redone whenever the normalizer
changes. Decomposes the edit distance, which is what lets the failure taxonomy be categorical
instead of threshold-driven, and scores against the closest accepted reading (SPEC 5.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rtb.textnorm import NORMALIZER_VERSION, expand_small_cardinals, normalize_text

METRIC_NAME = "wer"

# Operaciones del alineamiento, en el orden en que se recorre la referencia.
HIT, SUB, INS, DEL = "hit", "sub", "ins", "del"


@dataclass
class Alignment:
    """The edit distance, taken apart."""

    hits: int
    substitutions: int
    insertions: int
    deletions: int
    operations: list[str]
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def rate(self) -> float:
        """Errors over reference length — the WER (or CER) itself.

        Unbounded above by design: a model that emits three times the words scores over 1.0, and
        clipping that to 1.0 would hide exactly the failure worth seeing.
        """
        if self.reference_length == 0:
            return 0.0 if self.insertions == 0 else float(self.insertions)
        return self.errors / self.reference_length

    @property
    def insertion_rate(self) -> float:
        """Extra material relative to what was asked for. Scale-free, so one insertion on a
        one-word phrase is not treated like one on a thirty-word sentence."""
        return self.insertions / max(1, self.reference_length)

    @property
    def trailing_deletions(self) -> int:
        """Reference tokens missing from the END, with nothing matched after them.

        The structural signature of truncation: the model stopped early. Deletions in the middle
        are a different problem and are not counted here.
        """
        count = 0
        for operation in reversed(self.operations):
            if operation == DEL:
                count += 1
            elif operation == INS:
                continue  # una insercion no interrumpe la racha de faltantes al final
            else:
                break
        return count


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Levenshtein with a backtrace, so the operations are recoverable and not just the total."""
    rows, columns = len(reference), len(hypothesis)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0] = i
    for j in range(1, columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # sustitucion
                    cost[i][j - 1],  # insercion
                    cost[i - 1][j],  # borrado
                )

    operations: list[str] = []
    hits = substitutions = insertions = deletions = 0
    i, j = rows, columns
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and cost[i][j] == cost[i - 1][j - 1]
        ):
            operations.append(HIT)
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            operations.append(SUB)
            substitutions += 1
            i, j = i - 1, j - 1
        elif j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            operations.append(INS)
            insertions += 1
            j -= 1
        else:
            operations.append(DEL)
            deletions += 1
            i -= 1
    operations.reverse()
    return Alignment(hits, substitutions, insertions, deletions, operations, rows)


def max_ngram_repeats(words: Sequence[str], max_n: int = 4) -> int:
    """The longest run of a consecutively repeated n-gram, for n from 1 to `max_n`.

    Only *consecutive* repetition counts: a word that appears twice in a sentence is not a loop.
    """
    worst = 1
    for n in range(1, min(max_n, len(words)) + 1):
        index = 0
        while index + n <= len(words):
            gram = list(words[index : index + n])
            repeats = 1
            probe = index + n
            while probe + n <= len(words) and list(words[probe : probe + n]) == gram:
                repeats += 1
                probe += n
            worst = max(worst, repeats)
            index += 1
    return worst


def has_digits(text: str) -> bool:
    return any(character.isdigit() for character in text)


# Simbolos que se PRONUNCIAN pero que la normalizacion borra como puntuacion: si el ASR escribio
# uno, dijo palabras que la referencia tiene y la hipotesis normalizada perdio. Es el otro artefacto
# de formato, aparte de los digitos.
SPOKEN_SYMBOLS = "%$°"


def has_spoken_symbol(raw_text: str) -> bool:
    return any(symbol in raw_text for symbol in SPOKEN_SYMBOLS)


@dataclass
class Score:
    """WER and CER against the closest accepted reading."""

    wer: float
    cer: float
    words: Alignment
    characters: Alignment
    matched_form: str
    matched_index: int
    hypothesis: str
    glued: Alignment
    raw_hypothesis: str = ""

    @property
    def reference_words(self) -> int:
        return self.words.reference_length

    @property
    def glued_rate(self) -> float:
        """Character error with the spaces removed — how much material actually differs.

        A moved word boundary shows up at word level as a whole extra word. Taking the spaces out
        of the comparison leaves only the characters that genuinely differ: two, for a `llamalo`
        that came back as `y amalo`. That is what the boundary rule thresholds (SPEC 7.5).
        """
        return self.glued.rate

    @property
    def formatting_conflict(self) -> bool:
        """Whether the two sides disagree about *writing*, not about *speech*.

        Whisper renders spoken numbers back into digits, so the alignment would measure the ASR's
        convention instead of the model. Detected, not hardcoded per category (SPEC 7.6).
        """
        return has_digits(self.matched_form) != has_digits(self.hypothesis) or has_spoken_symbol(
            self.raw_hypothesis
        )


def score(reference_forms: Sequence[str], hypothesis: str) -> Score | None:
    """Score against every accepted form, keeping the closest.

    Chosen by CER and not WER, which is contaminated by number and acronym verbalization (SPEC 7.1).
    """
    hypothesis_normalized = expand_small_cardinals(normalize_text(hypothesis))
    hypothesis_words = hypothesis_normalized.split()

    best: Score | None = None
    for index, form in enumerate(reference_forms):
        reference_normalized = normalize_text(form)
        words = align(reference_normalized.split(), hypothesis_words)
        characters = align(list(reference_normalized), list(hypothesis_normalized))
        glued = align(
            list(reference_normalized.replace(" ", "")),
            list(hypothesis_normalized.replace(" ", "")),
        )
        candidate = Score(
            wer=words.rate,
            cer=characters.rate,
            words=words,
            characters=characters,
            matched_form=reference_normalized,
            matched_index=index,
            hypothesis=hypothesis_normalized,
            glued=glued,
            raw_hypothesis=hypothesis,
        )
        if best is None or candidate.cer < best.cer:
            best = candidate
    return best


def metric_version(asr_version: str) -> str:
    """Identity: these numbers move if the normalizer changes OR if the transcripts do.

    Both have to be in it. A WER tagged only with the normalizer would look current after the audio
    was re-transcribed with a different model.
    """
    return f"norm{NORMALIZER_VERSION}/{asr_version}"
