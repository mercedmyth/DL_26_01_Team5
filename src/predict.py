import torch
from PIL import Image
import pyttsx3

from .model import TrafficSignCNN
from .dataset import VAL_TRANSFORM, CLASS_NAMES

_tts_engine = None


def _get_tts():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", 160)
        # 한국어 음성이 있으면 설정 (시스템에 따라 다름)
        voices = _tts_engine.getProperty("voices")
        for v in voices:
            if "korean" in v.name.lower() or "ko" in v.id.lower():
                _tts_engine.setProperty("voice", v.id)
                break
    return _tts_engine


def load_model(checkpoint_path: str, device: torch.device) -> TrafficSignCNN:
    model = TrafficSignCNN(num_classes=43).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    image_path: str,
    checkpoint_path: str,
    speak: bool = True,
    top_k: int = 3,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    image = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(image).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    top_probs, top_indices = probs.topk(top_k)

    results = {
        "predicted_class": CLASS_NAMES[top_indices[0].item()],
        "confidence": top_probs[0].item(),
        "top_k": [
            {"class": CLASS_NAMES[idx.item()], "prob": prob.item()}
            for idx, prob in zip(top_indices, top_probs)
        ],
    }

    print(f"\n예측 결과: {results['predicted_class']} ({results['confidence']*100:.1f}%)")
    for i, r in enumerate(results["top_k"]):
        print(f"  Top-{i+1}: {r['class']}  ({r['prob']*100:.1f}%)")

    if speak:
        msg = f"인식된 표지판: {results['predicted_class']}"
        engine = _get_tts()
        engine.say(msg)
        engine.runAndWait()

    return results
