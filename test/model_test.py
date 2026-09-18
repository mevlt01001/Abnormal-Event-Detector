import pytest
import torch
from unittest.mock import patch, MagicMock

from model.model import model_creator
from model.model import Model, FC_head, create_base_model, get_feature_dim

true_model_names = model_creator.keys()

false_model_names = [
    "model",
    "123413",
    "model/",
    "model.py",
    "__init__.py",
    "__pycache__",
    "weights/",
    "test/",
    "asdsadasdsad/",
    "asdasd/asdasdasd",
]

model_fdims = {
    "mvit_v1_b": 768,
    "mvit_v2_s": 768,

    "r3d_18": 512,
    "mc3_18": 512,
    "r2plus1d_18": 512,

    "s3d": 1024,

    "swin3d_t": 768,
    "swin3d_s": 768,
    "swin3d_b": 1024,
}


def test_create_base_model():
    for model_name in ["r3d_18", "mc3_18"]:
        model = create_base_model(model_name)
        assert model.__class__.__name__ in [
            "MViT",
            "VideoResNet",
            "S3D",
            "SwinTransformer3d",
        ], f"Error: True model name {model_name} is not creating base model"

    for model_name in false_model_names:
        with pytest.raises(ValueError) as e:
            model = create_base_model(model_name)
        assert "There is no" in str(e.value), f"Error: False model name {model_name} is not raising ValueError"


def test_get_feature_dim():
    # Test on lightweight backbone to keep test fast
    model = create_base_model("r3d_18")
    feature_dim = get_feature_dim(model)
    assert feature_dim == 512, f"Expected 512, got {feature_dim}"


def test_fc_head():
    in_features = 512
    # Test default num_classes=1
    head = FC_head(in_features=in_features)

    linear_layers = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    expected_linear_count = 1 + 3 + 1  # 3 hidden layers by default
    assert len(linear_layers) == expected_linear_count

    dummy_x = torch.randn(4, in_features)
    head.eval()
    with torch.no_grad():
        out = head(dummy_x)
        assert out.shape == (4, 1), f"Expected output shape (4, 1), got {out.shape}"
        assert (out >= 0.0).all() and (out <= 1.0).all(), "Sigmoid output should be in range [0, 1]!"

        # Test L2 normalization: in eval mode, x and 100*x produce identical outputs
        out_scaled = head(dummy_x * 100.0)
        assert torch.allclose(out, out_scaled, atol=1e-5), "L2 normalization should make output scale-invariant!"

    # Test raw logits mode (use_sigmoid=False) for multi-class
    head_logits = FC_head(in_features=in_features, num_classes=6, use_sigmoid=False)
    head_logits.eval()
    with torch.no_grad():
        out_logits = head_logits(dummy_x)
        assert out_logits.shape == (4, 6)
        # Should not be restricted to [0, 1]



def test_model_forward():
    # Default instantiation without num_classes or hidden_dims
    model = Model(base_model_name="r3d_18")
    model.eval()

    with torch.no_grad():
        dummy_video = torch.randn(2, 3, 16, 224, 224)
        out = model(dummy_video)

    assert out.shape == (2, 1), f"Expected Model output shape: (2, 1), got: {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Model anomaly score should be in range [0, 1]!"


def test_model_extract_features_mock():
    model = Model(base_model_name="r3d_18")
    model.eval()

    # Mock fetch_video_segments to return 4 dummy segment tensors
    dummy_clips = [torch.randn(1, 3, 16, 224, 224) for _ in range(4)]

    with patch("utils.video_utils.fetch_video_segments", return_value=iter(dummy_clips)):
        feats = model.extract_features("fake_video.mp4", num_segments=4, batch_size=2)
        assert feats.shape == (4, 512), f"Expected shape (4, 512), got {feats.shape}"


def test_model_cuda_if_available():
    if torch.cuda.is_available():
        model = Model(base_model_name="r3d_18").to("cuda")
        dummy_video = torch.randn(1, 3, 16, 224, 224, device="cuda")
        out = model(dummy_video)
        assert out.device.type == "cuda"
        assert out.shape == (1, 1)
