# Prepared inputs

Place externally supplied, authorized inputs in the package root:

```text
data/<dataset>/
  features.npz          [N, F] sparse public item attributes
  text.npy              [N, T] frozen public-content vectors
  created.npy           [N] item-creation times
  address_features.npy  Yelp only: [city vocabulary, 6] fixed public features
  READY.json            input metadata
  D0/, D1/, D2/, D3/
    user_ids.npy        [U] stage-local user identities
    offsets.npy         [U+1] offsets into concatenated complete B events
    item.npy            concatenated B item IDs, preserving multiplicity
    timestamp.npy       timestamps aligned with item.npy
    round_user.npy      [G] owning user index
    round_index.npy     [G] zero-based personal group index
    targets.npy         [G, 5] sealed actual next-group items
    negatives.npy       [G, 15] fixed sampled training controls
    catalog.npy         stage candidate item IDs
    round_start.npy     [G] supplied group-start times
    round_end.npy       [G] supplied group-end times
    tags_clean.npy      [U, L] declaration IDs, padded with -1
    tags_deleted60.npy  fixed internal selection input
    tags_replaced60.npy fixed internal selection input
    address.npy         Yelp only: separately ordered activity-city tags
```

Item IDs index the same feature, text, and creation-time rows. Each user's B
contains whole groups of five; offsets, group ownership, and group indices must
agree. The two Yelp vocabularies are independent. Arrays named `address` refer
to activity cities, not residential addresses. Their six public features derive
from merchant latitude/longitude; the exact transform is a fixed input asset.
Declaration rows with no tags contain only -1.

The hidden earliest A used to synthesize MovieLens/Yelp declarations, ratings,
and virtual histories are not model inputs. No match matrices are needed by
this core release. Internal train/validation masks are recomputed from the
complete exported B timestamps, not loaded from older split arrays.

`configs/data_manifest.json` preserves the supplied fingerprints for retained
input assets. Checking them confirms equality to those fingerprints, not the
correctness of preprocessing. The historical full asset manifest also contains
baseline-only files; those are omitted here. An identical hash does not supply
the missing file. No automatic raw-data preprocessor is provided.

For MovieLens/Yelp, source preprocessing used the hidden earliest
`max(floor(0.2*n), 10)` events only for declarations, retained at least ten B
events, assigned disjoint user cohorts, and censored later cross-stage events.
Recreating these assets requires the frozen cohort/catalog mappings, content
vectors, declarations, selection perturbations, and sampled controls. The
distribution used to generate internal perturbation arrays is not specified
by their names or by this release.
