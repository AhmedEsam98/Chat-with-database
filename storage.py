"""
Azure Blob Storage module for managing schema backups, documents, and chat exports.

Handles container initialization, blob listing, uploading prompt schemas,
and downloading text content using azure-storage-blob.

Usage:
    from storage import is_storage_configured, backup_schema_to_blob, list_blobs, CONTAINER_NAME
"""

import asyncio
from datetime import datetime
from dotenv import load_dotenv
import config

load_dotenv()

CONNECTION_STRING = config.AZURE_STORAGE_CONNECTION_STRING
CONTAINER_NAME = config.AZURE_STORAGE_CONTAINER_NAME


def is_storage_configured() -> bool:
    """Check if Azure Blob Storage credentials are provided and valid."""
    return bool(config.AZURE_STORAGE_CONNECTION_STRING)


def _get_service_client():
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient.from_connection_string(config.AZURE_STORAGE_CONNECTION_STRING)


async def backup_schema_to_blob(schema_text: str) -> str | None:
    """Upload schema text to Azure Blob Storage as a timestamped markdown file."""
    if not is_storage_configured():
        return None

    def _upload():
        client = _get_service_client()
        container = client.get_container_client(CONTAINER_NAME)
        if not container.exists():
            container.create_container()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        blob_name = f"schemas/schema_{timestamp}.md"
        blob_client = container.get_blob_client(blob_name)
        blob_client.upload_blob(schema_text.encode("utf-8"), overwrite=True)
        return blob_client.url

    try:
        return await asyncio.to_thread(_upload)
    except Exception:
        return None


async def list_blobs() -> list[dict]:
    """List all blobs in the container."""
    if not is_storage_configured():
        return []

    def _list():
        client = _get_service_client()
        container = client.get_container_client(CONTAINER_NAME)
        if not container.exists():
            return []
        blobs = container.list_blobs()
        return [{"name": b.name, "size": b.size, "last_modified": str(b.last_modified)} for b in blobs]

    try:
        return await asyncio.to_thread(_list)
    except Exception:
        return []
