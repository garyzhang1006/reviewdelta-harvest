# Recorded prediction: cs.CV follow-up cell (cv25b)

Written before fetching any cv25b data, after observing the cv25 arm difference
of -0.144 (p=0.01, mean and rank tests) on 26 treatment / 64 control papers.

Cell: cs.CV, arXiv listing months 2025-01 and 2025-05, same harvest, arm, and
extraction pipeline as every other cell (replicate.py). All treatment papers
attempted plus seeded random controls to 100 diffs.

Decision rule, verbatim from the session log:
- If noise: pooled cv25+cv25b diff shrinks toward 0, p rises above 0.05
  (mirrors the first-tranche dissolve).
- If real: pooled diff stays at or below about -0.10 with p <= 0.01 at
  treatment n ~ 50+; then it enters the paper as a heterogeneity finding,
  not a multiplicity candidate.
- Either way the outcome is reported; no cherry-pick.

Outcome (recorded after the fetch): cv25b alone read -0.015 (p=0.81) on 15
treatment / 76 control. Pooled read -0.089 (p=0.022), between the two
registered branches and carried by the original cell. The paper reports it
as a failure to reproduce, per the replication-standard reading.

Limitation of this record: it is repository-hosted, not lodged with an
external registry, so its timing is attested by the release rather than by a
third party.
