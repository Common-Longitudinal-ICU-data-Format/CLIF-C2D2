"""Derive the adult-only C2D2 data dictionary from the official C2D2 Excel.

Run: uv run --with openpyxl --with pandas python scripts/make_adult_dictionary.py
"""

from pathlib import Path

import pandas as pd

DOCS = Path(__file__).resolve().parent.parent / "C2D2_docs"
SRC = DOCS / "ccm_53_4_2025_02_06_murphy_ccmed-d-24-01099_sdc2.xlsx"
OUT = DOCS / "c2d2_adult_data_dictionary.xlsx"
SHEET = "C2D2datadictionary"

df = pd.read_excel(SRC, sheet_name=SHEET)

# Pediatric rows: neonatal/infant elements (Condition "age < 2 years") and
# PRISM III pediatric-threshold variants (prism_*), which duplicate icu_24hr_* elements.
peds = df["Condition"].eq("age < 2 years") | df["Common Data Element"].str.startswith("prism_")
adult = df[~peds]

assert not adult["Condition"].eq("age < 2 years").any()
assert not adult["Common Data Element"].str.startswith("prism_").any()
assert len(adult) == 190, len(adult)

adult.to_excel(OUT, sheet_name=SHEET, index=False)
print(f"{len(df)} rows -> kept {len(adult)}, dropped {peds.sum()} -> {OUT.name}")
