import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def synthetic_corpus(tmp_path_factory):
    """Synthetic IEMOCAP-shaped corpus: (audio_root, manifest DataFrame with absolute paths)."""
    from emotisense.data.synthetic import build_synthetic_corpus

    root = tmp_path_factory.mktemp("synthetic")
    df = build_synthetic_corpus(root, dialogs_per_session=2, turns=6)
    return root, df
