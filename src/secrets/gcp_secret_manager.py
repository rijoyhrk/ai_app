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


def fetch_secret(project_id: str, secret_name: str, version: str = "latest") -> str:
    """
    Fetch a secret value from GCP Secret Manager.

    Args:
        project_id:  GCP project ID (e.g. "my-project-123")
        secret_name: Secret resource name (e.g. "anthropic-api-key")
        version:     Secret version — "latest" or a numeric version string

    Returns:
        The secret payload as a UTF-8 string.

    Raises:
        ImportError:   google-cloud-secret-manager is not installed.
        RuntimeError:  Secret could not be fetched (permission denied, not found, etc.).
    """
    try:
        from google.cloud import secretmanager  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-cloud-secret-manager is required to use GCP Secret Manager. "
            "Install it with:  pip install google-cloud-secret-manager"
        ) from exc

    resource = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    try:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": resource})
        return response.payload.data.decode("utf-8").strip()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch secret '{secret_name}' (version={version}) "
            f"from project '{project_id}': {exc}"
        ) from exc
