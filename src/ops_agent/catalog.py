"""Registers the six tools over the Online Retail II subset on a ToolRegistry."""

from __future__ import annotations

from typing import Any

from ops_agent.store import ORDER_SORTS, ORDER_STATUSES, Store
from ops_agent.tools import ToolRegistry, ToolSpec


def _obj(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _gbp(value: Any) -> str:
    return f"GBP {value:.2f}"


def _sum_search_customers(args: dict, d: dict) -> str:
    where = f" in {args['country']}" if args.get("country") else ""
    return f"{d['count']} customers{where}"


def _sum_get_customer(args: dict, d: dict) -> str:
    if not d.get("found"):
        return f"customer {d.get('customer_id')} not found"
    c = d["customer"]
    return f"customer {c['customer_id']} ({c['country']}), {d['order_count']} orders"


def _sum_get_order(args: dict, d: dict) -> str:
    if not d.get("found"):
        return f"order {d.get('order_id')} not found"
    o = d["order"]
    return f"order {o['order_id']} ({o['status']}), {o['line_count']} lines, {_gbp(o['total_gbp'])}"


def _sum_list_orders(args: dict, d: dict) -> str:
    if d.get("error"):
        return f"rejected: {d['error']}"
    return f"{d['count']} orders, total {_gbp(d['total_gbp'])}"


def _sum_get_product(args: dict, d: dict) -> str:
    if not d.get("found"):
        return f"no product matching {d.get('query')!r}"
    first = d["products"][0]
    if d["count"] == 1:
        return f"product {first['stock_code']}, {first['units_sold_in_window']} units sold in window"
    return f"{d['count']} products match, best {first['stock_code']}"


def _sum_create_ticket(args: dict, d: dict) -> str:
    if not d.get("created"):
        return f"ticket not created: {d.get('error')}"
    t = d["ticket"]
    return f"ticket {t['id']} created for customer {t['customer_id']}"


def build_registry(store: Store) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="search_customers",
        description=(
            "Find customers in the dataset subset, filtered by country and/or by part of the "
            "numeric customer id. Customers have only an id and a country (no names or contact "
            "details). Country names are the dataset's own, for example 'United Kingdom', "
            "'EIRE' (Ireland), 'USA', 'Channel Islands'; an unknown country returns "
            "known_countries. Each customer also carries orders_in_window, "
            "cancelled_orders_in_window and invoiced_total_gbp_in_window. Sort by customer_id "
            "(default), orders_desc or spend_desc. Returns up to `limit` customers (default 10, "
            "max 50) and the total match count."
        ),
        input_schema=_obj({
            "country": {"type": "string", "description": "Exact country name as in the dataset."},
            "query": {"type": "string", "description": "Digits contained in the customer id."},
            "sort_by": {
                "type": "string",
                "enum": ["customer_id", "orders_desc", "spend_desc"],
                "default": "customer_id",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
        }, []),
        fn=store.search_customers,
        summarize=_sum_search_customers,
    ))
    reg.register(ToolSpec(
        name="get_customer",
        description=(
            "Get one customer by numeric customer id (for example 12748): country, orders in "
            "the window, cancelled orders, total invoiced in GBP, and their most recent orders "
            "(up to 25, newest first; order_count is the full number). Returns found=false "
            "for an id that is not in this subset."
        ),
        input_schema=_obj({"customer_id": {"type": "string"}}, ["customer_id"]),
        fn=store.get_customer,
        summarize=_sum_get_customer,
    ))
    reg.register(ToolSpec(
        name="get_order",
        description=(
            "Get one order (an invoice) by invoice number (for example 565083; cancellation "
            "invoices start with C, for example C565085): date, customer, country, status "
            "(invoiced or cancelled), line_count, total in GBP and its lines with stock code, "
            "description, quantity, unit price and line total in GBP. Returns the first "
            "`max_lines` lines (default 25, max 100); line_count is the full number. Quantities "
            "and totals on cancelled orders are negative, as in the source data. Returns "
            "found=false for an unknown invoice number."
        ),
        input_schema=_obj({
            "order_id": {"type": "string"},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
        }, ["order_id"]),
        fn=store.get_order,
        summarize=_sum_get_order,
    ))
    reg.register(ToolSpec(
        name="list_orders",
        description=(
            "List orders (invoices) in the dataset subset, optionally filtered by customer_id, "
            "country, status (invoiced or cancelled), invoice date range (date_from, date_to as "
            "YYYY-MM-DD, inclusive) and minimum total in GBP. Sort by date_desc (default), "
            "date_asc, total_desc or total_asc. Returns the total match count, the total GBP "
            "value of all matches, counts_by_country and counts_by_status over ALL matches, and "
            "up to `limit` order summaries without lines (default 10, max 50)."
        ),
        input_schema=_obj({
            "customer_id": {"type": "string"},
            "country": {"type": "string", "description": "Exact country name as in the dataset."},
            "status": {"type": "string", "enum": list(ORDER_STATUSES)},
            "date_from": {"type": "string", "description": "YYYY-MM-DD"},
            "date_to": {"type": "string", "description": "YYYY-MM-DD"},
            "min_total_gbp": {"type": "number"},
            "sort": {"type": "string", "enum": list(ORDER_SORTS), "default": "date_desc"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
        }, []),
        fn=store.list_orders,
        summarize=_sum_list_orders,
    ))
    reg.register(ToolSpec(
        name="get_product",
        description=(
            "Look up product sales by stock code (exact, for example 85123A) or by words from "
            "the description (all words must match). There are NO stock levels in the dataset. "
            "Each product has its description, usual unit price in GBP and three figures counted "
            "over this subset only (not the whole retailer): units_sold_in_window and "
            "orders_in_window (invoiced orders only, cancellations not subtracted) and last_sold "
            "(date). Several matches are sorted by units sold, most first. Returns found=false "
            "when nothing matches."
        ),
        input_schema=_obj({
            "query": {"type": "string", "description": "Stock code or words from the description."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        }, ["query"]),
        fn=store.get_product,
        summarize=_sum_get_product,
    ))
    reg.register(ToolSpec(
        name="create_ticket",
        description=(
            "Create a support ticket for an existing customer id. Tickets are kept in memory "
            "only and are not part of the dataset. This writes data: state what you will "
            "create, then call with confirm=true."
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
        summarize=_sum_create_ticket,
    ))
    return reg
