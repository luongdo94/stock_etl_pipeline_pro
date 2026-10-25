"""Test helper: stamp synthetic snapshots with the CURRENT score and rules versions, as etl/snapshot.py does, so they
pass the evidence provenance gate (K4-08). Real snapshots always carry both stamps."""
from core import track_record as tr
from core.scoring import SCORE_VERSION


def stamped(snapshots):
    return snapshots.assign(score_version=SCORE_VERSION, rules_version=tr.scoring_definition() + "|test")
