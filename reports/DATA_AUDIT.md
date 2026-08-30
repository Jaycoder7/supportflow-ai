# Dataset Audit

## Source and license

The modeling data comes from the **Bitext Retail & E-commerce Customer Support
Dataset**, published under the CDLA-Sharing-1.0 license.

- Source: <https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset>
- Audited source revision: `12dd624ddcd3057382b2faad661bcda1fa869491`
- Raw records: 44,884
- Raw columns: `instruction`, `intent`, `category`, `tags`, `response`

The raw data is excluded from Git. The download script verifies the converted
parquet file's SHA-256 checksum before using it.

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

## Why this dataset was selected

An earlier candidate contained 8,469 tickets, but its categories and subjects
were largely independent of the message text. Trial models achieved only 0.177
validation macro-F1, so that source was rejected rather than used to report a
misleading result.

Bitext provides fine-grained intents that correspond directly to the customer
request. We aggregate a focused subset into the five business departments.

## Department mapping

| Source intents | Department |
| --- | --- |
| `pay`, `payment_issue`, `payment_methods` | Billing |
| `technical_issue`, `product_issue` | Technical Support |
| `recover_password` | Account Access |
| `refund_policy`, `refund_status`, `request_refund` | Refund |
| `submit_feedback`, `submit_product_feedback`, `submit_product_idea` | Product Feedback |

## Leakage and privacy controls

- The model receives only the customer's cleaned `instruction`.
- `intent` and `category` construct the target but are never model features.
- `response` is excluded because it is produced after the customer asks for
  support.
- Generation `tags` are excluded because users do not submit those values.
- Emails, URLs, long numeric identifiers, and template entity placeholders are
  normalized.
- Duplicate cleaned requests are removed before splitting.

## Known limitations

- The messages are generated examples, not production tickets.
- The dataset includes deliberately varied tones, including profanity.
- Account Access currently represents password recovery, not every possible
  account-access failure.
- Results measure performance on this dataset and may not transfer directly to
  a particular company's incoming tickets.
- The dataset does not supply operational urgency labels. Priority will use a
  documented rule system instead of an unsupported learned classifier.
