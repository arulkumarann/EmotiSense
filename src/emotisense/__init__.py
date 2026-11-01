"""EmotiSense: second-by-second speech emotion timelines."""
import warnings

# Raised inside Hugging Face's WavLM attention on recent PyTorch; harmless and very noisy.
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")

__version__ = "0.1.0"
