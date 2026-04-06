from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .services import (
    get_training_log_tail,
    get_training_status,
    infer_uploaded_files,
    save_uploaded_files,
    start_training_process,
    stop_training_process,
)


def _bad_request(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"ok": False, "message": message}, status=status)


@require_GET
def health(request):
    return JsonResponse({"ok": True, "message": "backend is running"})


@csrf_exempt
@require_POST
def test_archive_api(request):
    files = request.FILES.getlist("files") or ([request.FILES["file"]] if "file" in request.FILES else [])
    if not files:
        return _bad_request("请通过 files 或 file 上传至少一个音频文件")

    invalid = [f.name for f in files if Path(f.name).suffix.lower() not in settings.ALLOWED_AUDIO_EXTENSIONS]
    if invalid:
        return _bad_request(f"存在不支持的文件格式: {invalid}")

    try:
        results = infer_uploaded_files(files)
        return JsonResponse({"ok": True, "count": len(results), "results": results})
    except FileNotFoundError:
        return _bad_request(f"模型文件不存在: {settings.MODEL_PATH}", status=500)
    except Exception as exc:
        return _bad_request(f"推理失败: {exc}", status=500)


@csrf_exempt
@require_POST
def train_api(request):
    try:
        result = start_training_process()
        code = 200 if result.get("started") else 409
        return JsonResponse({"ok": result.get("started", False), **result}, status=code)
    except Exception as exc:
        return _bad_request(f"启动训练失败: {exc}", status=500)


@require_GET
def train_status_api(request):
    try:
        status = get_training_status()
        return JsonResponse({"ok": True, **status})
    except Exception as exc:
        return _bad_request(f"查询训练状态失败: {exc}", status=500)


@csrf_exempt
@require_POST
def train_stop_api(request):
    try:
        result = stop_training_process()
        code = 200 if result.get("stopped") else 409
        return JsonResponse({"ok": result.get("stopped", False), **result}, status=code)
    except Exception as exc:
        return _bad_request(f"停止训练失败: {exc}", status=500)


@require_GET
def train_log_api(request):
    raw_n = request.GET.get("n", "100")
    try:
        n = int(raw_n)
    except ValueError:
        return _bad_request("参数 n 必须是整数")

    try:
        result = get_training_log_tail(n)
        return JsonResponse({"ok": True, **result})
    except Exception as exc:
        return _bad_request(f"读取训练日志失败: {exc}", status=500)


@csrf_exempt
@require_POST
def archive_upload_api(request):
    label = request.POST.get("label", "").strip().lower()
    if label not in {"fake", "real"}:
        return _bad_request("label 必须为 fake 或 real")

    files = request.FILES.getlist("files") or ([request.FILES["file"]] if "file" in request.FILES else [])
    if not files:
        return _bad_request("请通过 files 或 file 上传至少一个音频文件")

    invalid = [f.name for f in files if Path(f.name).suffix.lower() not in settings.ALLOWED_AUDIO_EXTENSIONS]
    if invalid:
        return _bad_request(f"存在不支持的文件格式: {invalid}")

    try:
        saved = save_uploaded_files(label, files)
        return JsonResponse({"ok": True, "count": len(saved), "saved": saved})
    except Exception as exc:
        return _bad_request(f"保存失败: {exc}", status=500)
