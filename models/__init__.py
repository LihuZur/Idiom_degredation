"""Model registry, runners and device helper (ARCHITECTURE §2.5).

Current phase: HF-only, decoder runners only.
"""

from models import (
    decoder_runner as decoder_runner,
)
from models import (
    mistral7b_instruct_v03 as mistral7b_instruct_v03,
)
from models import (
    qwen3p5_1p5b_instruct as qwen3p5_1p5b_instruct,
)
