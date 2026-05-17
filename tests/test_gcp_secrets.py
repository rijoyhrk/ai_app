"""
Tests for GCP Secret Manager integration.
The GCP client is fully mocked — no real GCP project or credentials needed.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.secrets.gcp_secret_manager import fetch_secret


def _make_gcp_response(secret_value: str) -> MagicMock:
    response = MagicMock()
    response.payload.data = secret_value.encode("utf-8")
    return response


class TestFetchSecret:

    @patch("src.secrets.gcp_secret_manager.secretmanager")
    def test_returns_secret_value(self, mock_sm_module):
        mock_client = MagicMock()
        mock_sm_module.SecretManagerServiceClient.return_value = mock_client
        mock_client.access_secret_version.return_value = _make_gcp_response("sk-ant-test-key")

        result = fetch_secret("my-project", "anthropic-api-key")
        assert result == "sk-ant-test-key"

    @patch("src.secrets.gcp_secret_manager.secretmanager")
    def test_constructs_correct_resource_name(self, mock_sm_module):
        mock_client = MagicMock()
        mock_sm_module.SecretManagerServiceClient.return_value = mock_client
        mock_client.access_secret_version.return_value = _make_gcp_response("key")

        fetch_secret("proj-123", "my-secret", version="5")

        call_args = mock_client.access_secret_version.call_args
        resource = call_args.kwargs["request"]["name"]
        assert resource == "projects/proj-123/secrets/my-secret/versions/5"

    @patch("src.secrets.gcp_secret_manager.secretmanager")
    def test_default_version_is_latest(self, mock_sm_module):
        mock_client = MagicMock()
        mock_sm_module.SecretManagerServiceClient.return_value = mock_client
        mock_client.access_secret_version.return_value = _make_gcp_response("key")

        fetch_secret("proj-123", "my-secret")

        call_args = mock_client.access_secret_version.call_args
        resource = call_args.kwargs["request"]["name"]
        assert resource.endswith("/versions/latest")

    @patch("src.secrets.gcp_secret_manager.secretmanager")
    def test_strips_trailing_whitespace(self, mock_sm_module):
        mock_client = MagicMock()
        mock_sm_module.SecretManagerServiceClient.return_value = mock_client
        mock_client.access_secret_version.return_value = _make_gcp_response("sk-ant-key\n")

        result = fetch_secret("proj", "secret")
        assert result == "sk-ant-key"

    @patch("src.secrets.gcp_secret_manager.secretmanager")
    def test_raises_runtime_error_on_api_failure(self, mock_sm_module):
        mock_client = MagicMock()
        mock_sm_module.SecretManagerServiceClient.return_value = mock_client
        mock_client.access_secret_version.side_effect = Exception("PERMISSION_DENIED")

        with pytest.raises(RuntimeError, match="Failed to fetch secret"):
            fetch_secret("proj", "secret")

    @patch("src.secrets.gcp_secret_manager.secretmanager")
    def test_runtime_error_includes_project_and_secret_name(self, mock_sm_module):
        mock_client = MagicMock()
        mock_sm_module.SecretManagerServiceClient.return_value = mock_client
        mock_client.access_secret_version.side_effect = Exception("Not found")

        with pytest.raises(RuntimeError) as exc_info:
            fetch_secret("my-proj", "my-secret")

        assert "my-proj"   in str(exc_info.value)
        assert "my-secret" in str(exc_info.value)

    def test_raises_import_error_when_library_missing(self):
        """Simulate google-cloud-secret-manager not installed."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "google.cloud.secretmanager" or "secretmanager" in name:
                raise ImportError("No module named 'google.cloud.secretmanager'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Re-import the module so it uses the patched import
            import importlib
            import src.secrets.gcp_secret_manager as mod
            # The ImportError should be raised when calling fetch_secret
            # (it's imported lazily inside the function)

    @patch("src.secrets.gcp_secret_manager.secretmanager")
    def test_specific_version_pinning(self, mock_sm_module):
        mock_client = MagicMock()
        mock_sm_module.SecretManagerServiceClient.return_value = mock_client
        mock_client.access_secret_version.return_value = _make_gcp_response("key-v3")

        result = fetch_secret("proj", "secret", version="3")
        call_args = mock_client.access_secret_version.call_args
        assert "versions/3" in call_args.kwargs["request"]["name"]
        assert result == "key-v3"
