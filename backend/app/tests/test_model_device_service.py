from backend.app.services.model_device_service import (
    force_cpu_fallback,
    get_model_device_info,
    is_cuda_failure,
    reset_runtime_device_fallback_for_tests,
)


def test_model_device_info_reports_selected_device() -> None:
    reset_runtime_device_fallback_for_tests()
    info = get_model_device_info()

    assert info.selected_device in {"cpu", "cuda"}
    assert info.requested_device in {"auto", "cpu", "cuda"}
    assert isinstance(info.cuda_available, bool)
    if info.selected_device == "cuda":
        assert info.cuda_available is True
        assert info.cuda_device_count >= 1


def test_cuda_failure_detection_covers_oom() -> None:
    assert is_cuda_failure(RuntimeError("CUDA out of memory"))
    assert not is_cuda_failure(RuntimeError("plain network failure"))


def test_runtime_cpu_fallback_is_recorded() -> None:
    reset_runtime_device_fallback_for_tests()
    force_cpu_fallback("test cuda oom")

    info = get_model_device_info()

    assert info.selected_device == "cpu"
    assert info.fallback_reason == "test cuda oom"
    reset_runtime_device_fallback_for_tests()
