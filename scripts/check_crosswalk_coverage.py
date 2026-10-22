"""Check that the domain crosswalk YAMLs hold exactly the adult C2D2 CDEs.

Run: uv run --with openpyxl --with pandas --with pyyaml python scripts/check_crosswalk_coverage.py
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
# Required keys per source_type (see README "CDE labels")
SOURCE_TYPES = {
    "direct": {"clif_table", "clif_column", "output_type"},
    "derived": {"clif_table", "clif_category_column", "clif_category_value", "clif_value_column",
                "clif_time_column", "aggregation", "time_window", "output_type"},
    "pass_through_ontology": {"clif_table", "code_format", "code_prefixes", "output_type"},
    "custom": {"clif_tables", "implementation", "output_type"},
    "unmapped": {"clif_status", "reason"},
}
CLIF_STATUSES = {"clif_gap", "clif_partial", "not_yet_mapped", "external_data"}

excel = set(pd.read_excel(ROOT / "C2D2_docs" / "c2d2_adult_data_dictionary.xlsx")["Common Data Element"])
names, mapped = [], Counter()
for f in sorted((ROOT / "src" / "crosswalks").glob("*.yaml")):
    entries = yaml.safe_load(f.read_text()) or []
    for e in entries:
        assert e["source_type"] in SOURCE_TYPES, (f.name, e)
        assert SOURCE_TYPES[e["source_type"]] <= e.keys(), (f.name, e["c2d2_name"], SOURCE_TYPES[e["source_type"]] - e.keys())
        assert e["source_type"] != "unmapped" or e["clif_status"] in CLIF_STATUSES, (f.name, e)
        names.append(e["c2d2_name"])
    mapped[f.stem] = sum(e["source_type"] != "unmapped" for e in entries)
    print(f"{f.stem:24} {mapped[f.stem]:3} / {len(entries)} mapped")

assert len(names) == len(set(names)), [n for n, c in Counter(names).items() if c > 1]
assert set(names) == excel, {"missing": excel - set(names), "extra": set(names) - excel}
print(f"OK: {len(names)} CDEs, {sum(mapped.values())} mapped")
