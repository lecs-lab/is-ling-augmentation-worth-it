"""
Implements a variation of chrF, where word order is completely ignored.

We do this by using the sacrebleu CHRF implementation, but removing any n-grams that span word boundaries.
"""

from collections import Counter

from sacrebleu.metrics.chrf import CHRF

chrf = CHRF(word_order=0)


def free_word_chrf(hypotheses: list[str], references: list[str]):
    """Computes a variant of chrF, where word order is entirely disregarded."""
    assert len(hypotheses) == len(references)
    CHAR_ORDER = 6

    def _extract_char_grams(sentence: str):
        words = sentence.split()
        counters = []
        for n in range(1, CHAR_ORDER + 1):
            # This is the key modification:
            # Since we get n-grams separately for each word, we won't have any
            # n-grams with spaces in them.
            ngrams = Counter(
                [word[i : i + n] for word in words for i in range(len(word) - n + 1)]
            )
            counters.append(ngrams)
        return counters

    all_stats = []
    for pred, ref in zip(hypotheses, references):
        hypothesis_grams = _extract_char_grams(chrf._preprocess_segment(pred))
        reference_grams = _extract_char_grams(chrf._preprocess_segment(ref))

        stats = []
        # Traverse all orders
        for h, r in zip(hypothesis_grams, reference_grams):
            stats.extend(chrf._get_match_statistics(h, r))
        all_stats.append(stats)
    score = chrf._aggregate_and_compute(all_stats)
    return score


refs = ["the big dog runs up the street.", "my name is michael"]

preds = ["bic tha do runs the streeet. up", "name is michael my"]

print(free_word_chrf(preds, refs))
print(chrf.corpus_score(preds, [refs]))
