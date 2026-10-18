"""In-memory data store over the JSON files in data/.

Every method returns a plain JSON-serialisable dict. Unknown ids return
{"found": False} instead of raising. Tickets are held in memory and, when a
tickets_path is given, persisted there; data/ is never written.
"""

from __future__ import annotations

import json
import pathlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

MAX_SUBJECT_CHARS = 200
MAX_BODY_CHARS = 4000
ORDER_STATUSES = frozenset({"placed", "shipped", "delivered", "cancelled"})


class Customer(BaseModel):
    id: str
    name: str
    email: str
    city: str
    state: str
    tier: str


class OrderItem(BaseModel):
    sku: str
    qty: int
    unit_price_usd: float


class Order(BaseModel):
    id: str
    customer_id: str
    status: str
    items: list[OrderItem]
    total_usd: float
    placed_on: str


class InventoryItem(BaseModel):
    sku: str
    name: str
    on_hand: int
    reorder_point: int
    warehouse: str


class Ticket(BaseModel):
    id: str
    customer_id: str
    subject: str
    body: str
    created_at: str


def _load(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


class Store:
    def __init__(self, data_dir: pathlib.Path, tickets_path: pathlib.Path | None = None):
        data_dir = pathlib.Path(data_dir)
        self._customers = {c["id"]: Customer(**c) for c in _load(data_dir / "customers.json")}
        self._orders = {o["id"]: Order(**o) for o in _load(data_dir / "orders.json")}
        self._inventory = {i["sku"]: InventoryItem(**i) for i in _load(data_dir / "inventory.json")}
        self._tickets_path = pathlib.Path(tickets_path) if tickets_path else None
        source = self._tickets_path or data_dir / "tickets.json"
        self._tickets = [Ticket(**t) for t in _load(source)]

    def search_customers(self, query: str, limit: int = 10) -> dict[str, Any]:
        limit = max(1, min(50, int(limit)))
        q = str(query).strip().lower()
        hits = [
            c
            for c in self._customers.values()
            if q and (q in c.name.lower() or q in c.email.lower())
        ]
        return {
            "count": len(hits),
            "customers": [c.model_dump() for c in hits[:limit]],
        }

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        c = self._customers.get(customer_id)
        if c is None:
            return {"found": False, "customer_id": customer_id}
        orders = [o.id for o in self._orders.values() if o.customer_id == customer_id]
        return {"found": True, "customer": c.model_dump(), "order_ids": orders}

    def get_order(self, order_id: str) -> dict[str, Any]:
        o = self._orders.get(order_id)
        if o is None:
            return {"found": False, "order_id": order_id}
        return {"found": True, "order": o.model_dump()}

    def list_orders(self, status: str | None = None, limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(50, int(limit)))
        if status is not None and status not in ORDER_STATUSES:
            return {
                "count": 0,
                "orders": [],
                "error": f"status must be one of {sorted(ORDER_STATUSES)}",
            }
        rows = [o for o in self._orders.values() if status is None or o.status == status]
        return {
            "count": len(rows),
            "orders": [o.model_dump() for o in rows[:limit]],
        }

    def check_inventory(self, sku: str) -> dict[str, Any]:
        i = self._inventory.get(sku)
        if i is None:
            return {"found": False, "sku": sku}
        return {
            "found": True,
            "item": i.model_dump(),
            "in_stock": i.on_hand > 0,
            "below_reorder_point": i.on_hand <= i.reorder_point,
        }

    def create_ticket(self, customer_id: str, subject: str, body: str) -> dict[str, Any]:
        if customer_id not in self._customers:
            return {"created": False, "error": "unknown customer", "customer_id": customer_id}
        if not subject.strip() or len(subject) > MAX_SUBJECT_CHARS:
            return {"created": False, "error": f"subject must be 1-{MAX_SUBJECT_CHARS} characters"}
        if len(body) > MAX_BODY_CHARS:
            return {"created": False, "error": f"body must be at most {MAX_BODY_CHARS} characters"}
        ticket = Ticket(
            id=f"T-{9001 + len(self._tickets)}",
            customer_id=customer_id,
            subject=subject,
            body=body,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._tickets.append(ticket)
        if self._tickets_path is not None:
            self._tickets_path.write_text(
                json.dumps([t.model_dump() for t in self._tickets], indent=2) + "\n"
            )
        return {"created": True, "ticket": ticket.model_dump()}
