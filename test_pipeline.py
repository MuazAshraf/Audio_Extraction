"""
Test file to verify spectrogram + CNN encoder pipeline works.
"""
import torch
from asr.features import get_spectrogram_for_cnn
from cnn_encoder import EncoderCNN

AUDIO_PATH = "libri_sample.wav"

def test_with_real_audio():
    """Test with real audio file"""
    print("=" * 50)
    print("Testing with REAL audio...")
    print("=" * 50)
    
    # Step 1: Get spectrogram from audio
    print(f"Loading audio: {AUDIO_PATH}")
    spec = get_spectrogram_for_cnn(AUDIO_PATH)
    print(f"Spectrogram shape: {spec.shape}")  # Should be (128, T)
    
    # Step 2: Prepare for CNN (add batch and channel dims)
    spec = spec.unsqueeze(0).unsqueeze(0)  # (1, 1, 128, T)
    print(f"After unsqueeze: {spec.shape}")
    
    # Step 3: Pass through CNN encoder
    encoder = EncoderCNN()
    output = encoder(spec)
    print(f"CNN output shape: {output.shape}")  # Should be (1, T/8, 512)
    print("✓ Real audio test PASSED!")


if __name__ == "__main__":
    test_with_real_audio()
