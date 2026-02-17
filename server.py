"""MCP server that connects Notion to the RAG memory system."""

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server.lowlevel import NotificationOptions, Server
import mcp.server.stdio
from mcp.server.models import InitializationOptions
from mcp.types import TextContent, Tool

from config import ServerConfig, NotionConfig, SyncConfig
from notion_client import NotionClient
from notion_sync import NotionSyncer
from rag_adapter import NotionRAGAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("notion-mcp")

# --- Server setup ---

server = Server("notion-rag-memory")
config = ServerConfig()
adapter = NotionRAGAdapter(index_path=config.sync.index_path)


# --- Tool definitions ---

TOOLS = [
    Tool(
        name="sync_notion",
        description=(
            "Sync Notion pages and databases into the local RAG index. "
            "Supports full sync (rebuild all) and incremental sync (only changed pages). "
            "Provide page_ids and/or database_ids to specify what to sync."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Notion page IDs to sync",
                },
                "database_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Notion database IDs to sync",
                },
                "mode": {
                    "type": "string",
                    "enum": ["full", "incremental"],
                    "default": "incremental",
                    "description": "Sync mode: 'full' rebuilds all, 'incremental' syncs only changes",
                },
            },
        },
    ),
    Tool(
        name="search_notion_memory",
        description=(
            "Search over indexed Notion content using semantic similarity. "
            "Returns the most relevant chunks matching the query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results to return",
                },
                "threshold": {
                    "type": "number",
                    "default": 0.0,
                    "description": "Minimum similarity score (0.0-1.0)",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="ask_notion",
        description=(
            "Ask a question and get an answer based on indexed Notion content. "
            "Uses the full RAG pipeline: retrieve relevant chunks, "
            "augment the prompt, and generate an answer with citations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of chunks to retrieve",
                },
                "threshold": {
                    "type": "number",
                    "default": 0.3,
                    "description": "Similarity threshold for filtering chunks",
                },
                "enforce_citations": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to require citations in the answer",
                },
            },
            "required": ["question"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "sync_notion":
            return await _handle_sync(arguments)
        elif name == "search_notion_memory":
            return await _handle_search(arguments)
        elif name == "ask_notion":
            return await _handle_ask(arguments)
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )]
    except Exception as e:
        logger.exception("Error handling tool call: %s", name)
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}),
        )]


# --- Tool handlers ---

async def _handle_sync(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle the sync_notion tool call."""
    page_ids = arguments.get("page_ids", [])
    database_ids = arguments.get("database_ids", [])
    mode = arguments.get("mode", "incremental")

    if not page_ids and not database_ids:
        # Use configured defaults
        page_ids = config.sync.page_ids
        database_ids = config.sync.database_ids

    if not page_ids and not database_ids:
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": "No page_ids or database_ids provided. "
                "Specify them in the tool call or set them in config.",
            }),
        )]

    syncer = NotionSyncer(config)

    if mode == "full":
        result = await syncer.full_sync(
            page_ids=page_ids or None,
            database_ids=database_ids or None,
        )
    else:
        result = await syncer.incremental_sync(
            page_ids=page_ids or None,
            database_ids=database_ids or None,
        )

    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "success",
            "mode": mode,
            **result.to_dict(),
        }, indent=2),
    )]


async def _handle_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle the search_notion_memory tool call."""
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)
    threshold = arguments.get("threshold", 0.0)

    if not query:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "Query is required"}),
        )]

    results = adapter.search(query=query, top_k=top_k, threshold=threshold)

    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": [r.to_dict() for r in results],
        }, indent=2),
    )]


async def _handle_ask(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle the ask_notion tool call."""
    question = arguments.get("question", "")
    top_k = arguments.get("top_k", 5)
    threshold = arguments.get("threshold", 0.3)
    enforce_citations = arguments.get("enforce_citations", False)

    if not question:
        return [TextContent(
            type="text",
            text=json.dumps({"error": "Question is required"}),
        )]

    result = adapter.ask(
        question=question,
        top_k=top_k,
        threshold=threshold,
        enforce_citations=enforce_citations,
    )

    return [TextContent(
        type="text",
        text=json.dumps({
            "status": "success",
            **result,
        }, indent=2),
    )]


# --- Entry point ---

async def main() -> None:
    logger.info("Starting Notion RAG MCP server...")
    logger.info("Index path: %s", config.sync.index_path)

    if not config.notion.api_key:
        logger.error("NOTION_API_KEY environment variable is not set")
        sys.exit(1)

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="notion-rag-memory",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
