# SupportFlow AI Model Card

## Model summary

SupportFlow AI is a five-class English customer-support router. The selected
model is a scikit-learn pipeline containing word-level TF-IDF unigrams and
bigrams followed by class-balanced multinomial logistic regression.

Model version: `2026-08-30T01:46:12+00:00`

## Intended use

- Assist with first-pass routing of English retail and e-commerce support requests
- Surface uncertain tickets for human review
- Demonstrate interpretable classical NLP in an educational portfolio project

The system is decision support. A support organization remains responsible for
the final route and response.

## Out-of-scope use

- Eligibility, employment, lending, insurance, medical, or legal decisions
- Fraud adjudication or security-incident containment
- Fully autonomous processing of sensitive or high-impact requests
- Languages or business domains not evaluated here

## Departments

- Billing
- Technical Support
- Account Access
- Refund
- Product Feedback

## Training data

Source: Bitext Retail & E-commerce Customer Support Dataset

- License: CDLA-Sharing-1.0
- Audited source revision: `12dd624ddcd3057382b2faad661bcda1fa869491`
- Processed records: 11,823
- Features used: cleaned customer `instruction` only
- Target source: selected fine-grained intent labels aggregated into departments

Agent responses, source categories, intent strings, and generation tags are
excluded from features. See [DATA_AUDIT.md](DATA_AUDIT.md) for mapping and
privacy controls.

## Evaluation procedure

The cleaned, deduplicated data uses a fixed, stratified split:

- Training: 8,276 tickets (70%)
- Validation: 1,773 tickets (15%)
- Test: 1,774 tickets (15%)

Logistic regression and multinomial Naive Bayes used identical TF-IDF settings
and splits. The model was selected by validation macro-F1. The test set was
evaluated after selecting the model and review threshold.

## Results

| Metric | Logistic regression | Naive Bayes |
| --- | ---: | ---: |
| Validation accuracy | 0.998 | 0.993 |
| Validation macro-F1 | 0.999 | 0.994 |

Selected-model test results:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.998 |
| Macro-F1 | 0.998 |
| Auto-routing coverage at threshold 0.80 | 0.960 |
| Accuracy among auto-routed tickets | 1.000 |
| Human-review rate | 0.040 |

All four misclassified test examples fell below the 0.80 threshold and were
sent to human review. Detailed machine-readable results are in `metrics.json`;
the review sample is in `error_analysis.csv`.

## Confidence and human review

The classifier exposes `predict_proba`, but its output is a model score—not a
guarantee. Candidate thresholds were evaluated on the validation set for
selective accuracy, auto-routing coverage, and review rate. The 0.80 threshold
was selected to stay near a 5% review budget while maintaining at least 99.5%
validation accuracy among accepted tickets.

Probability calibration was not performed. Any production pilot should assess
calibration and reselect the threshold using real operational costs.

## Explanations

For the predicted class, the interface calculates:

```text
ticket TF-IDF value × learned logistic-regression coefficient
```

It displays the strongest positive contributions among words and phrases found
in the submitted ticket. These values describe learned associations and are not
causal explanations.

## Priority

Priority is not learned from this dataset because it contains no operational
urgency labels. A documented rule policy assigns Critical, High, Medium, or Low
and shows matched phrases. This policy requires business-owner review before
production use.

## Limitations and risks

- The evaluation text is generated and strongly templated, explaining the
  unusually high score.
- Performance on real messages, longer conversations, code snippets, or mixed
  intents is unknown.
- Account Access represents password recovery and has less semantic coverage
  than the other departments.
- Mixed billing/refund requests can be ambiguous; the confidence policy is
  essential.
- Deliberate language variation includes profanity and misspellings.
- Word features may learn spurious dataset-specific phrasing.
- Feedback CSV storage is not durable on ephemeral cloud deployments.

## Monitoring and retraining recommendations

- Track category volume, review rate, correction rate, and per-class recall
- Sample both high- and low-confidence predictions for human audit
- Alert on shifts in word distribution or confidence distribution
- Version all reviewed corrections and retain their provenance
- Keep evaluation tickets out of future training sets
- Require a new model card and threshold analysis for each model version
