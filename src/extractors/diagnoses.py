"""Domain 4: Diagnoses - ~30 ICD10 diagnosis flags."""

import polars as pl
from .base import BaseExtractor


class DiagnosesExtractor(BaseExtractor):
    crosswalk_file = "diagnoses.yaml"

    def extract(self) -> pl.DataFrame:
        return self._extract_pass_through_ontology()
