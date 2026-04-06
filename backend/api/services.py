import os
import re
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Any
from typing import cast

import soundfile as sf
import torch
import torchaudio
from django.conf import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import LightweightCNN

CONFIDENCE_THRESHOLD = 0.8
GROUP_DIR_PATTERN = re.compile(r"^group_(\d{3})$")


class InferenceEngine:
    """线程安全的惰性模型加载器。"""

    def __init__(self) -> None:
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = Lock()

    def _load_model(self) -> None:
        model = LightweightCNN(num_classes=2)
        checkpoint = torch.load(settings.MODEL_PATH, map_location=self._device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(self._device)
        model.eval()
        self._model = model

    def get_model(self) -> LightweightCNN:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
        return cast(LightweightCNN, self._model)

    @staticmethod
    def _load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
        try:
            waveform, sr = torchaudio.load(audio_path)
            return waveform.float(), sr
        except Exception:
            audio, sr = sf.read(audio_path)
            waveform = torch.from_numpy(audio).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.transpose(0, 1)
            return waveform, sr

    @staticmethod
    def preprocess_audio(audio_path: str, sample_rate: int = 16000, duration: int = 2) -> torch.Tensor:
        target_len = duration * sample_rate
        waveform, sr = InferenceEngine._load_audio(audio_path)

        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
        )
        mel = mel_transform(waveform)
        mel = torchaudio.transforms.AmplitudeToDB()(mel)

        if mel.std() > 0:
            mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        return mel.unsqueeze(0)

    def predict(self, audio_path: str) -> dict[str, Any]:
        model = self.get_model()
        mel = self.preprocess_audio(audio_path).to(self._device)

        with torch.no_grad():
            outputs = model(mel)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)

        pred_idx = int(pred.item())
        confidence = float(probs[0][pred_idx].item())
        result = "AI合成" if pred_idx == 1 else "真人"
        if confidence < CONFIDENCE_THRESHOLD:
            result = "AI合成"

        return {
            "result": result,
            "confidence": confidence,
            "pred_idx": pred_idx,
        }


INFERENCE_ENGINE = InferenceEngine()


def _list_group_dirs(root: Path) -> list[Path]:
    def _group_index(path: Path) -> int:
        match = GROUP_DIR_PATTERN.match(path.name)
        return int(match.group(1)) if match else 0

    group_dirs = [p for p in root.iterdir() if p.is_dir() and GROUP_DIR_PATTERN.match(p.name)]
    group_dirs.sort(key=_group_index)
    return group_dirs


def _audio_count(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in settings.ALLOWED_AUDIO_EXTENSIONS)


def _next_group_name(existing: list[Path]) -> str:
    if not existing:
        return "group_001"
    max_idx = max(
        int(match.group(1))
        for p in existing
        for match in [GROUP_DIR_PATTERN.match(p.name)]
        if match
    )
    return f"group_{max_idx + 1:03d}"


def get_or_create_group_dir(label: str) -> Path:
    label_root = settings.DATA_ROOT / label
    label_root.mkdir(parents=True, exist_ok=True)

    groups = _list_group_dirs(label_root)
    for group_dir in groups:
        if _audio_count(group_dir) < settings.MAX_AUDIO_FILES_PER_GROUP:
            return group_dir

    new_group = label_root / _next_group_name(groups)
    new_group.mkdir(parents=True, exist_ok=True)
    return new_group


def save_uploaded_files(label: str, files: list[Any]) -> list[dict[str, str]]:
    saved: list[dict[str, str]] = []
    remaining = list(files)

    while remaining:
        target_group = get_or_create_group_dir(label)
        can_put = settings.MAX_AUDIO_FILES_PER_GROUP - _audio_count(target_group)
        batch = remaining[:can_put]
        remaining = remaining[can_put:]

        for upload in batch:
            suffix = Path(upload.name).suffix.lower()
            unique_name = f"{Path(upload.name).stem}_{uuid.uuid4().hex[:8]}{suffix}"
            save_path = target_group / unique_name
            with open(save_path, "wb+") as out:
                for chunk in upload.chunks():
                    out.write(chunk)
            saved.append(
                {
                    "filename": upload.name,
                    "saved_as": unique_name,
                    "label": label,
                    "group": target_group.name,
                    "path": str(save_path.relative_to(settings.DATA_ROOT)),
                }
            )

    return saved


def infer_uploaded_files(files: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for upload in files:
        suffix = Path(upload.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in upload.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            pred = INFERENCE_ENGINE.predict(tmp_path)
            pred["filename"] = upload.name
            results.append(pred)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return results


def start_training_process() -> dict[str, Any]:
    pid_file = Path(__file__).resolve().parents[1] / "train.pid"
    log_file = Path(__file__).resolve().parents[1] / "train.log"

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
            return {"started": False, "message": "训练任务已在运行中", "pid": old_pid}
        except Exception:
            pid_file.unlink(missing_ok=True)

    with open(log_file, "a", encoding="utf-8") as log_fp:
        process = subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "train.py")],
            cwd=str(PROJECT_ROOT),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )

    pid_file.write_text(str(process.pid), encoding="utf-8")
    return {"started": True, "message": "训练任务已启动", "pid": process.pid, "log": str(log_file)}


def _train_pid_file() -> Path:
    return Path(__file__).resolve().parents[1] / "train.pid"


def _train_log_file() -> Path:
    return Path(__file__).resolve().parents[1] / "train.log"


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        # On some Windows setups this still indicates the process exists.
        return True
    except OSError:
        return False


def get_training_status() -> dict[str, Any]:
    pid_file = _train_pid_file()
    log_file = _train_log_file()

    if not pid_file.exists():
        return {
            "running": False,
            "pid": None,
            "message": "当前无训练任务",
            "log": str(log_file),
        }

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        pid_file.unlink(missing_ok=True)
        return {
            "running": False,
            "pid": None,
            "message": "PID 文件损坏，已清理",
            "log": str(log_file),
        }

    if _is_pid_running(pid):
        return {
            "running": True,
            "pid": pid,
            "message": "训练任务运行中",
            "log": str(log_file),
        }

    pid_file.unlink(missing_ok=True)
    return {
        "running": False,
        "pid": pid,
        "message": "训练任务已结束",
        "log": str(log_file),
    }


def stop_training_process() -> dict[str, Any]:
    status = get_training_status()
    if not status.get("running"):
        return {
            "stopped": False,
            "pid": status.get("pid"),
            "message": "当前没有可停止的训练任务",
            "log": status.get("log"),
        }

    pid = int(status["pid"])

    if os.name == "nt":
        # Kill process tree to avoid orphan dataloader workers.
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        success = completed.returncode == 0
        detail = completed.stdout.strip() or completed.stderr.strip()
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            success = True
            detail = ""
        except Exception as exc:
            success = False
            detail = str(exc)

    if success:
        _train_pid_file().unlink(missing_ok=True)

    return {
        "stopped": success,
        "pid": pid,
        "message": "训练任务已停止" if success else "停止训练任务失败",
        "detail": detail,
        "log": status.get("log"),
    }


def _tail_lines(file_path: Path, n: int) -> list[str]:
    if n <= 0:
        return []

    with open(file_path, "rb") as fp:
        fp.seek(0, os.SEEK_END)
        pointer = fp.tell()
        chunk_size = 4096
        buffer = b""
        line_count = 0

        while pointer > 0 and line_count <= n:
            read_size = min(chunk_size, pointer)
            pointer -= read_size
            fp.seek(pointer)
            data = fp.read(read_size)
            buffer = data + buffer
            line_count = buffer.count(b"\n")

        text = buffer.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-n:]


def get_training_log_tail(lines: int = 100) -> dict[str, Any]:
    safe_lines = max(1, min(lines, 2000))
    log_file = _train_log_file()

    if not log_file.exists():
        return {
            "exists": False,
            "log": str(log_file),
            "lines": safe_lines,
            "content": [],
            "message": "训练日志文件不存在",
        }

    content = _tail_lines(log_file, safe_lines)
    return {
        "exists": True,
        "log": str(log_file),
        "lines": safe_lines,
        "content": content,
        "message": "ok",
    }
