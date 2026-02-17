"""Configuration for the Notion MCP server."""

import os
from dataclasses import dataclass, field


@dataclass
class NotionConfig:
    """Configuration for Notion API access."""
    api_key: str = field(
        default_factory=lambda: os.environ.get("NOTION_API_KEY", "")
    )
    api_version: str = "2022-06-28"
    base_url: str = "https://api.notion.com/v1"
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class SyncConfig:
    """Configuration for Notion sync behavior."""
    index_path: str = "notion_index.json"
    sync_state_path: str = "notion_sync_state.json"
    page_ids: list[str] = field(default_factory=list)
    database_ids: list[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    """Combined configuration for the MCP server."""
    notion: NotionConfig = field(default_factory=NotionConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
