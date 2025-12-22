import torch
from cnn_encoder import EncoderCNN, Seq2SeqASR

def test_pipeline():
    print("=" * 50)
    print("Testing with DUMMY data...")
    print("=" * 50)
    
    # Dummy spectrogram (Batch=2, Channel=1, Freq=128, Time=400)
    spec = torch.randn(2, 1, 128, 400)
    print(f"Input shape: {spec.shape}")
    
    # Test CNN Encoder
    encoder = EncoderCNN()
    encoder_output = encoder(spec)
    print(f"CNN Encoder output: {encoder_output.shape}")
    
    # Test Full Seq2Seq Model
    model = Seq2SeqASR(vocab_size=29)
    target_text = torch.randint(0, 29, (2, 20))  # Dummy target
    output = model(spec, target_text)
    print(f"Seq2Seq output: {output.shape}")
    
    print("=" * 50)
    print("✓ All tests PASSED!")
    print("=" * 50)

if __name__ == "__main__":
    test_pipeline()
