from django.urls import path

from .views import (
    archive_upload_api,
    health,
    test_archive_api,
    train_api,
    train_log_api,
    train_status_api,
    train_stop_api,
)

urlpatterns = [
    path("health/", health, name="health"),
    path("test-archive/", test_archive_api, name="test_archive_api"),
    path("train/", train_api, name="train_api"),
    path("train/status/", train_status_api, name="train_status_api"),
    path("train/stop/", train_stop_api, name="train_stop_api"),
    path("train/log/", train_log_api, name="train_log_api"),
    path("archive/upload/", archive_upload_api, name="archive_upload_api"),
]
