# McNemar significance report — mistral-7b-instruct-v0.3

## What is being compared

Restricting to the 1548 items where this model answered
the **original** variant correctly, we compare whether the **paraphrase** or the **idiomatic**
rewrite of that same item broke the answer, while the other rewrite did not:

|                        | idiomatic WRONG | idiomatic right |
|------------------------|------------------|------------------|
| **paraphrase WRONG**   | (both broke it — excluded from the test) | c = 80 |
| **paraphrase right**   | b = 32 | (both fine — excluded from the test) |

Only the *discordant* cells (b, c) — where exactly one rewrite broke an otherwise-correct
answer — enter McNemar's test. The concordant cells (both right / both wrong) carry no
information about which rewrite is worse, so they are excluded, as in any paired McNemar test.
For reference, `original_only_wrong` = 51 (cases where the original itself was wrong
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
- c (paraphrase-only-wrong) = 80
- n = b + c = 112
- chi2 = 19.723, df = 1
- p-value = 0.00001

## Conclusion

**Statistically significant at alpha=0.05** (p=0.00001 < 0.05): the **paraphrase** rewrite broke a previously-correct answer more often than the **idiomatic** rewrite did (c=80 vs b=32). This is the OPPOSITE of the research hypothesis (paraphrase degraded accuracy more than idiomatic here).
