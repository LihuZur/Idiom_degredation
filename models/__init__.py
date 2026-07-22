"""Model registry, runners and device helper (ARCHITECTURE §2.5).

Current phase: HF-only, decoder runners only.
"""

from models import (
    decoder_runner as decoder_runner,
)
from models import (
    deepseek_r1_distill_qwen_7b as deepseek_r1_distill_qwen_7b,
)
from models import (
    gemma4_9b_instruct as gemma4_9b_instruct,
)
from models import (
    mistral7b_instruct_v03 as mistral7b_instruct_v03,
)
from models import (
    phi4_mini_instruct as phi4_mini_instruct,
)
from models import (
    qwen3p5_1p5b_instruct as qwen3p5_1p5b_instruct,
)
from models import (
    qwen3p5_7b_instruct as qwen3p5_7b_instruct,
)
