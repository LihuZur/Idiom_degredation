"""SST-2 loader tests — hits the real HF hub (STAGE1_PLAN §5.5, Q17)."""

import re

from data.sst2 import Sst2Loader

_HF_REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def test_sst2_loader_yields_expected_rows() -> None:
    loader = Sst2Loader(hf_revision=_HF_REVISION, normalize=["nfc", "collapse_whitespace"])
    rows = list(loader.load())

    assert len(rows) > 60_000
    assert all(row.y != -1 for row in rows)
    assert all(_ID_RE.match(row.id) for row in rows)
