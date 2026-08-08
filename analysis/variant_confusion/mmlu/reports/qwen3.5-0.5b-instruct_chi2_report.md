# McNemar significance report — qwen3.5-0.5b-instruct

## What is being compared

Restricting to the 1085 items where this model answered
the **original** variant correctly, we compare whether the **paraphrase** or the **idiomatic**
rewrite of that same item broke the answer, while the other rewrite did not:

|                        | idiomatic WRONG | idiomatic right |
|------------------------|------------------|------------------|
| **paraphrase WRONG**   | (both broke it — excluded from the test) | c = 19 |
| **paraphrase right**   | b = 32 | (both fine — excluded from the test) |

Only the *discordant* cells (b, c) — where exactly one rewrite broke an otherwise-correct
answer — enter McNemar's test. The concordant cells (both right / both wrong) carry no
information about which rewrite is worse, so they are excluded, as in any paired McNemar test.
For reference, `original_only_wrong` = 16 (cases where the original itself was wrong
but both rewrites were correct) is reported separately and does not enter this test.

## Test statistic

McNemar's test (with continuity correction) is:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

**Null hypothesis H0:** b and c come from the same underlying rate — i.e. a paraphrase rewrite
and an idiomatic rewrite are equally likely to break an originally-correct answer (each
discordant item is effectively a coin flip, p=0.5, for "idiomatic broke it" vs "paraphrase
broke it").

### Where the formula comes from (variance derivation)

Under H0, with n = b + c discordant items held fixed, the count B (of those n items assigned to
"idiomatic broke it") follows a Binomial(n, 0.5) distribution. Then:

$$\mathrm{Var}(B) = n \cdot 0.5 \cdot (1 - 0.5) = n / 4$$

Since C = n - B, the difference B - C = 2B - n, so:

$$\mathrm{Var}(B - C) = 4 \cdot \mathrm{Var}(B) = n$$

The (uncorrected) test statistic $z = (B - C) / \sqrt{\mathrm{Var}(B - C)} = (B - C) /
\sqrt{n}$ is then approximately standard normal under H0, and $z^2 \sim \chi^2(1)$. The
"-1" in $|b - c| - 1$ is Yates' continuity correction, which compensates for approximating a
discrete binomial difference with a continuous chi-square distribution — a standard adjustment
for paired 2x2 designs, especially with small counts.

## Result

- b (idiomatic-only-wrong) = 32
- c (paraphrase-only-wrong) = 19
- n = b + c = 51
- chi2 = 2.824, df = 1
- p-value = 0.09289

## Non-parametric bootstrap check

McNemar's test above relies on a chi-square approximation, which can be inaccurate when the
discordant count (b + c) is small. As a robustness check, we instead **resample the aligned
items with replacement** 10000 times (paired bootstrap — each resample redraws whole
`(original, paraphrase, idiomatic)` rows, keeping the pairing intact), recomputing
`idiomatic_only_wrong - paraphrase_only_wrong` on each resample. This builds an empirical
sampling distribution with no assumption of normality/chi-square, from which we read off a 95%
percentile confidence interval and a two-sided empirical p-value (the fraction of resamples
landing on the opposite side of zero from the observed difference, doubled).

- observed diff (b - c) = 13
- 95% bootstrap CI = [-1, 27]
- bootstrap p-value = 0.08220 (n_boot=10000)

## Conclusion

**Not statistically significant at alpha=0.05** (p=0.09289 >= 0.05). Although the **idiomatic** rewrite broke a previously-correct answer more often than the **paraphrase** rewrite did (b=32 vs c=19), with only 51 discordant items this difference could plausibly be due to chance alone — no reliable conclusion should be drawn from this result in isolation.

The bootstrap check is **more cautious**: its 95% CI on (idiomatic_only minus paraphrase_only) is [-1, 27] (includes zero), and its empirical p-value is 0.08220. This is broadly consistent with the McNemar result above.
