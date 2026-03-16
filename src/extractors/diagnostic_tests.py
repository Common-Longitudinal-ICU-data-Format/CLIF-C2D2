"""Domain 5: Diagnostic Tests - labs max/min in first 24h."""

import polars as pl
from .base import BaseExtractor


class DiagnosticTestsExtractor(BaseExtractor):
    crosswalk_file = "diagnostic_tests.yaml"

    def extract(self) -> pl.DataFrame:
        return self._extract_derived()
