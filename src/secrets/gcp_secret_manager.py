"""
GCP Secret Manager integration.

On a GCP VM (or any GCP compute resource) the attached service account is
used automatically via Application Default Credentials — no credential file
or hardcoded key is needed.

For local development, run:
    gcloud auth application-default login
or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON key file.
"""
from __future__ import annotations

import concurrent.futures

# Module-level import so tests can patch `src.secrets.gcp_secret_manager.secretmanager`.
# Gracefully set to None when the library is not installed; fetch_secret raises a
# clear ImportError in that case rather than at import time.
try:
    from google.cloud import secretmanager  # type: ignore
except ImportError:
    secretmanager = None  # type: ignore


def fetch_secret(
    project_id: str,
    secret_name: str,
    version: str = "latest",
    timeout: float = 5.0,
) -> str:
    """
    Fetch a secret value from GCP Secret Manager with a hard timeout.

    Args:
        project_id:  GCP project ID (e.g. "my-project-123")
        secret_name: Secret resource name (e.g. "anthropic-api-key")
        version:     Secret version — "latest" or a numeric version string
        timeout:     Seconds to wait before raising RuntimeError (default 5)

    Returns:
        The secret payload as a UTF-8 string.

    Raises:
        ImportError:   google-cloud-secret-manager is not installed.
        RuntimeError:  Secret could not be fetched (timeout, permission denied, not found, etc.).
    """
    if secretmanager is None:
        raise ImportError(
            "google-cloud-secret-manager is required to use GCP Secret Manager. "
            "Install it with:  pip install google-cloud-secret-manager"
        )

    resource = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"

    def _fetch() -> str:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": resource})
        return response.payload.data.decode("utf-8").strip()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise RuntimeError(
            f"GCP Secret Manager timed out after {timeout}s fetching '{secret_name}' "
            f"from project '{project_id}'. Check network connectivity and IAM permissions."
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch secret '{secret_name}' (version={version}) "
            f"from project '{project_id}': {exc}"
        ) from exc
