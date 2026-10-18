"""Registers the six Lumen Outdoor Co. tools on a ToolRegistry."""

from __future__ import annotations

from ops_agent.store import Store
from ops_agent.tools import ToolRegistry, ToolSpec


def _obj(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_registry(store: Store) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="search_customers",
        description=(
            "Find customers whose name or email contains the query text "
            "(case-insensitive). Returns up to `limit` matches (default 10, max 50) "
            "and the total match count."
        ),
        input_schema=_obj({
            "query": {"type": "string", "description": "Text to match in name or email."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
        }, ["query"]),
        fn=store.search_customers,
    ))
    reg.register(ToolSpec(
        name="get_customer",
        description=(
            "Get one customer by id (for example C-1001), including the ids of "
            "their orders. Returns found=false for an unknown id."
        ),
        input_schema=_obj({"customer_id": {"type": "string"}}, ["customer_id"]),
        fn=store.get_customer,
    ))
    reg.register(ToolSpec(
        name="get_order",
        description=(
            "Get one order by id (for example O-5001): status, items, total in USD "
            "and order date. Returns found=false for an unknown id."
        ),
        input_schema=_obj({"order_id": {"type": "string"}}, ["order_id"]),
        fn=store.get_order,
    ))
    reg.register(ToolSpec(
        name="list_orders",
        description=(
            "List orders, optionally filtered by status (placed, shipped, delivered "
            "or cancelled). Returns up to `limit` orders (default 20, max 50) and the "
            "total count."
        ),
        input_schema=_obj({
            "status": {
                "type": "string",
                "enum": ["placed", "shipped", "delivered", "cancelled"],
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
        }, []),
        fn=store.list_orders,
    ))
    reg.register(ToolSpec(
        name="check_inventory",
        description=(
            "Check stock for one SKU (for example LOB-TNT-2P): on-hand units, reorder "
            "point and warehouse. Returns found=false for an unknown SKU."
        ),
        input_schema=_obj({"sku": {"type": "string"}}, ["sku"]),
        fn=store.check_inventory,
    ))
    reg.register(ToolSpec(
        name="create_ticket",
        description=(
            "Create a support ticket for an existing customer. This writes data: "
            "state what you will create, then call with confirm=true."
        ),
        input_schema=_obj({
            "customer_id": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "confirm": {
                "type": "boolean",
                "description": "Must be true to perform the write.",
            },
        }, ["customer_id", "subject", "body", "confirm"]),
        fn=store.create_ticket,
        write=True,
    ))
    return reg
