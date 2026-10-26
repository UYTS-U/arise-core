# Core protocol

## Declarations and personal feedback

Each nonempty declaration group contributes fixed mass 0.7; an empty group
contributes zero. Ordered groups use exponential decay with gamma 0.35;
unordered groups use uniform coefficients. Groups have separate prototypes
and TL modules and sum without a second normalization: two nonempty groups
contribute total mass 1.4. Feedback never updates these coefficients.

The encoder reads actual revealed B, with long history 100. TL supplies the
zero-behavior residual. CWM retains the source module's two default outputs
for behavior and CT mixing; short history and short CT share W in {10,20,50},
and long CT uses 50. Endpoint gain EMA is strictly lagged. A whole five-event
group is predicted before any of its feedback is revealed. Shared weights and
statistics remain frozen during service; personal history and lagged evidence
advance. No teacher or distillation is used.

## Cohort boundaries

D0 fits the shared model. D1 and D2 each serve first, select block epochs on
the current cohort's temporal validation, reset model and RNG to stage entry,
and refit authorized current-cohort data. D3 only serves. Shared updates read
the current cohort, retained parameters, and sufficient statistics, without
replaying previous raw cohorts.

For each internal split, the cutoff is the 80% order statistic of exported
complete B timestamps. Training groups end before the cutoff; validation
groups start at or after it. Crossing groups are excluded from losses and
metrics, while revealed feedback can enter later contexts. D0 allows 100
epochs per block; D1/D2 selection allows 20, minimum actual epochs 3 and
patience 2. The selected checkpoint can precede epoch 3.

Native selection averages Clean/Replaced60 first/follow NDCG@10. TL uses
weighted Clean/Replaced60 first-round objectives; CWM uses Clean/Delete60
follow-up NDCG@5/@10. Exact weights are in `training.objective`. Formal service
uses Clean only. D0 training counts initialize CT before scale/coefficient/window
selection. Later reads retain prior CT statistics through refit, then write
the current increment exactly once. D0 selects lambda_I and W; later service
scores do not reselect them.

ML-1M removals use the same-seed ARISE reference's D0 settings and retrain
remaining blocks. Removing CT sets lambda_I to zero. TL first-round treatments
and CWM/CT follow-up treatments answer different questions.

## Study-specific interpretation

| Study | Declaration structure | Visibility and scope |
| --- | --- | --- |
| MovieLens-1M/10M | One ordered Genre group | Supplied recorded-stage roster assumption; revealed B masked |
| ZhihuRec | One ordered group, 106 followed-Topic clusters | Supplied group-start visibility; Topic-clustering and snapshot chronology unverified |
| Bluesky A | One unordered group, 11 untimed Feed selections | Sealed group's last timestamp; retrospective hidden-group recovery, outside strict prospective information assumptions |
| Yelp | Two independently ordered interest/activity-city groups | Supplied public-B business roster including tails; revisits allowed |

These studies should be interpreted separately. No online deployment or
verified registration-time availability is established by this code.

## Metrics

NDCG, Recall, and IDCG use unique within-group targets, while history/counts
retain event multiplicity. Target sets are not silently filtered against
candidate eligibility. Follow-up first averages within each user and then
across users; D1–D3 pooling weights users. The core emits run metrics, without
manufacturing extra trials, significance stars, or cross-study averages.
The sampled training controls are not observed negative feedback.
