# Error Analysis

The untouched test set contains 1,774 tickets. Logistic regression made four
errors, and every error had confidence below the 0.80 automation threshold.

| True department | Predicted department | Confidence | Observation |
| --- | --- | ---: | --- |
| Billing | Refund | 0.338 | Very short checkout wording; route was ambiguous. |
| Refund | Billing | 0.430 | Misspelled “reimbursement” weakened the refund signal. |
| Billing | Refund | 0.470 | Payment-method request contained heavy misspellings. |
| Technical Support | Product Feedback | 0.550 | “Product issue” phrasing overlaps feedback vocabulary. |

The review threshold behaved as intended: it prevented all four mistakes from
being auto-routed. The accompanying `error_analysis.csv` includes these errors
plus the lowest-confidence correct predictions, for a total of 20 review cases.

## Likely improvements

- Add character-level n-grams to improve robustness to misspellings.
- Collect mixed-intent examples such as billing requests that also ask for a refund.
- Expand technical examples that mention generic “product issues.”
- Evaluate the threshold again on real tickets rather than generated text.

