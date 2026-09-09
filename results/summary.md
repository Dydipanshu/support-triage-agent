# Results on golden set (n=220)

| system | intent acc | macro F1 | esc. recall | esc. precision | auto rate | unsafe auto | judge overall | share post-as-is |
|---|---|---|---|---|---|---|---|---|
| trivial | 0.177 | 0.033 | 1.000 | 0.232 | 0.000 | 0 | 1.98 | 0.236 |
| simple | 0.582 | 0.528 | 0.333 | 0.531 | 0.855 | 34 | 1.52 | 0.077 |
| agent | 0.841 | 0.839 | 0.941 | 0.649 | 0.664 | 3 | 2.39 | 0.455 |

## trivial: per-intent F1
| intent | precision | recall | f1 | support |
|---|---|---|---|---|
| praise_or_share | 0.00 | 0.00 | 0.00 | 33 |
| flight_disruption | 0.00 | 0.00 | 0.00 | 31 |
| booking_or_seat_change | 0.00 | 0.00 | 0.00 | 32 |
| baggage_or_lost_item | 0.00 | 0.00 | 0.00 | 16 |
| loyalty_or_account | 0.00 | 0.00 | 0.00 | 16 |
| policy_or_info_question | 0.00 | 0.00 | 0.00 | 22 |
| service_complaint_or_feedback | 0.18 | 1.00 | 0.30 | 39 |
| support_access_issue | 0.00 | 0.00 | 0.00 | 16 |
| other | 0.00 | 0.00 | 0.00 | 15 |

## simple: per-intent F1
| intent | precision | recall | f1 | support |
|---|---|---|---|---|
| praise_or_share | 0.59 | 0.82 | 0.68 | 33 |
| flight_disruption | 0.95 | 0.68 | 0.79 | 31 |
| booking_or_seat_change | 0.69 | 0.56 | 0.62 | 32 |
| baggage_or_lost_item | 0.78 | 0.44 | 0.56 | 16 |
| loyalty_or_account | 0.73 | 0.50 | 0.59 | 16 |
| policy_or_info_question | 0.36 | 0.45 | 0.40 | 22 |
| service_complaint_or_feedback | 0.55 | 0.59 | 0.57 | 39 |
| support_access_issue | 0.39 | 0.88 | 0.54 | 16 |
| other | 0.00 | 0.00 | 0.00 | 15 |

## agent: per-intent F1
| intent | precision | recall | f1 | support |
|---|---|---|---|---|
| praise_or_share | 0.97 | 0.97 | 0.97 | 33 |
| flight_disruption | 0.76 | 0.94 | 0.84 | 31 |
| booking_or_seat_change | 0.89 | 0.78 | 0.83 | 32 |
| baggage_or_lost_item | 1.00 | 0.94 | 0.97 | 16 |
| loyalty_or_account | 0.76 | 0.81 | 0.79 | 16 |
| policy_or_info_question | 0.75 | 0.95 | 0.84 | 22 |
| service_complaint_or_feedback | 0.89 | 0.64 | 0.75 | 39 |
| support_access_issue | 0.72 | 0.81 | 0.76 | 16 |
| other | 0.80 | 0.80 | 0.80 | 15 |
