"""In-memory data store over the JSON files in data/.

The data is a subset of the UCI "Online Retail II" dataset (real transactions of a UK online
retailer, see data/SOURCE.md). Customers are only a numeric id and a country. Orders are
invoices: a sales invoice has status "invoiced", a cancellation invoice (number starts with "C")
has status "cancelled". Money is GBP. The dataset has no stock levels, no delivery status and
no personal data, so none of those exist here.

Every method returns a plain JSON-serialisable dict. Unknown ids return {"found": False}
instead of raising. Fields named `*_in_window` are computed here from the loaded invoices.
Tickets are held in memory and, when a tickets_path is given, persisted there; data/ is never
written.
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from pydantic import BaseModel

MAX_SUBJECT_CHARS = 200
MAX_BODY_CHARS = 4000
ORDER_STATUSES = ("invoiced", "cancelled")
CUSTOMER_SORTS = ("customer_id", "orders_desc", "spend_desc")
ORDER_SORTS = ("date_desc", "date_asc", "total_desc", "total_asc")
MAX_ORDERS_PER_CUSTOMER_LISTING = 25
_CENT = Decimal("0.01")


class Ticket(BaseModel):
    id: str
    customer_id: str
    subject: str
    body: str
    created_at: str


def _load(path: pathlib.Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _cents(price: float, qty: int) -> int:
    """Line total in pence, rounded half up, from the price as printed in the data."""
    return int((Decimal(str(price)) * qty / _CENT).quantize(Decimal(1), ROUND_HALF_UP))


def _gbp(cents: int) -> float:
    return cents / 100


def _clamp(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def norm_id(value: Any) -> str:
    """Customer and order ids are strings in the data; accept ints and '12748.0' from callers."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


class Store:
    def __init__(
        self,
        data_dir: pathlib.Path,
        tickets_path: pathlib.Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        # Injectable so the eval runner can fix created_at; ticket ids come from the ticket
        # count, so they are deterministic whenever the starting tickets file is.
        self._clock = clock or (lambda: datetime.now(UTC))
        data_dir = pathlib.Path(data_dir)
        meta = _load(data_dir / "meta.json")
        self.meta: dict[str, Any] = meta if isinstance(meta, dict) else {}
        self._customers: dict[str, dict[str, Any]] = {
            c["id"]: dict(c) for c in _load(data_dir / "customers.json")
        }
        self._products: dict[str, dict[str, Any]] = {
            p["stock_code"].upper(): dict(p) for p in _load(data_dir / "products.json")
        }
        self._orders: dict[str, dict[str, Any]] = {}
        self._build_orders(_load(data_dir / "orders.json"))
        self._derive()
        self._tickets_path = pathlib.Path(tickets_path) if tickets_path else None
        source = self._tickets_path or data_dir / "tickets.json"
        self._tickets = [Ticket(**t) for t in _load(source)]

    # ---- loading -------------------------------------------------------------------------

    def _build_orders(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            lines = [(code, int(q), float(p), _cents(float(p), int(q))) for code, q, p in r["lines"]]
            self._orders[r["id"]] = {
                "id": r["id"],
                "invoice_date": r["date"],
                "customer_id": r["customer_id"],
                "country": r["country"],
                "status": r["status"],
                "lines": lines,
                "total_cents": sum(line[3] for line in lines),
            }

    def _derive(self) -> None:
        """Per-customer and per-product figures over the loaded invoices."""
        stats: dict[str, dict[str, Any]] = {
            cid: {"orders": 0, "cancelled": 0, "cents": 0} for cid in self._customers
        }
        sold: dict[str, dict[str, Any]] = {}
        for o in self._orders.values():
            s = stats.get(o["customer_id"])
            if s is None:
                continue
            if o["status"] == "cancelled":
                s["cancelled"] += 1
                continue
            s["orders"] += 1
            s["cents"] += o["total_cents"]
            for code, qty, _price, _cents_ in o["lines"]:
                p = sold.setdefault(code.upper(), {"units": 0, "orders": set(), "last": ""})
                p["units"] += qty
                p["orders"].add(o["id"])
                p["last"] = max(p["last"], o["invoice_date"][:10])
        for cid, s in stats.items():
            self._customers[cid].update(
                orders_in_window=s["orders"],
                cancelled_orders_in_window=s["cancelled"],
                invoiced_total_gbp_in_window=_gbp(s["cents"]),
            )
        for code, p in self._products.items():
            s = sold.get(code)
            p["units_sold_in_window"] = s["units"] if s else 0
            p["orders_in_window"] = len(s["orders"]) if s else 0
            p["last_sold"] = s["last"] if s else None
        self._countries = sorted({c["country"] for c in self._customers.values()})

    # ---- dataset facts -------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            "customers": len(self._customers),
            "orders": len(self._orders),
            "products": len(self._products),
            "order_lines": sum(len(o["lines"]) for o in self._orders.values()),
        }

    def dataset_info(self) -> dict[str, Any]:
        m = self.meta
        return {
            "name": m.get("name"),
            "source_url": m.get("source_url"),
            "license": m.get("license"),
            "doi": m.get("doi"),
            "date_pulled": m.get("date_pulled"),
            "window": m.get("window"),
            "counts": self.counts(),
            "currency": m.get("currency", "GBP"),
        }

    def examples(self) -> list[str]:
        return [e["question"] for e in self.meta.get("examples", [])]

    # ---- views ---------------------------------------------------------------------------

    def _customer_view(self, c: dict[str, Any]) -> dict[str, Any]:
        return {
            "customer_id": c["id"],
            "country": c["country"],
            "orders_in_window": c["orders_in_window"],
            "cancelled_orders_in_window": c["cancelled_orders_in_window"],
            "invoiced_total_gbp_in_window": c["invoiced_total_gbp_in_window"],
        }

    def _order_summary(self, o: dict[str, Any]) -> dict[str, Any]:
        return {
            "order_id": o["id"],
            "invoice_date": o["invoice_date"],
            "customer_id": o["customer_id"],
            "country": o["country"],
            "status": o["status"],
            "line_count": len(o["lines"]),
            "total_gbp": _gbp(o["total_cents"]),
        }

    def _product_view(self, p: dict[str, Any]) -> dict[str, Any]:
        return {
            "stock_code": p["stock_code"],
            "description": p["description"],
            "unit_price_gbp": p["unit_price_gbp"],
            "units_sold_in_window": p["units_sold_in_window"],
            "orders_in_window": p["orders_in_window"],
            "last_sold": p["last_sold"],
        }

    # ---- tools ---------------------------------------------------------------------------

    def search_customers(
        self,
        country: str | None = None,
        query: str | None = None,
        sort_by: str = "customer_id",
        limit: int = 10,
    ) -> dict[str, Any]:
        limit = _clamp(limit, 1, 50, 10)
        if sort_by not in CUSTOMER_SORTS:
            return {"count": 0, "customers": [], "error": f"sort_by must be one of {list(CUSTOMER_SORTS)}"}
        rows = list(self._customers.values())
        if country is not None and str(country).strip():
            want = str(country).strip().lower()
            rows = [c for c in rows if c["country"].lower() == want]
            if not rows:
                return {"count": 0, "customers": [], "known_countries": self._countries}
        if query is not None and str(query).strip():
            q = norm_id(query)
            rows = [c for c in rows if q in c["id"]]
        if sort_by == "orders_desc":
            rows.sort(key=lambda c: (-c["orders_in_window"], c["id"]))
        elif sort_by == "spend_desc":
            rows.sort(key=lambda c: (-c["invoiced_total_gbp_in_window"], c["id"]))
        else:
            rows.sort(key=lambda c: (len(c["id"]), c["id"]))
        return {"count": len(rows), "customers": [self._customer_view(c) for c in rows[:limit]]}

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        cid = norm_id(customer_id)
        c = self._customers.get(cid)
        if c is None:
            return {"found": False, "customer_id": cid}
        mine = sorted(
            (o for o in self._orders.values() if o["customer_id"] == cid),
            key=lambda o: (o["invoice_date"], o["id"]),
            reverse=True,
        )
        shown = mine[:MAX_ORDERS_PER_CUSTOMER_LISTING]
        return {
            "found": True,
            "customer": self._customer_view(c),
            "orders": [
                {k: s[k] for k in ("order_id", "invoice_date", "status", "total_gbp")}
                for s in map(self._order_summary, shown)
            ],
            "order_count": len(mine),
            "orders_listed": len(shown),
        }

    def get_order(self, order_id: str, max_lines: int = 25) -> dict[str, Any]:
        oid = str(order_id).strip().upper()
        o = self._orders.get(oid)
        if o is None:
            return {"found": False, "order_id": str(order_id).strip()}
        max_lines = _clamp(max_lines, 1, 100, 25)
        lines = o["lines"][:max_lines]
        order = self._order_summary(o)
        order["lines"] = [
            {
                "stock_code": code,
                "description": self._products.get(code.upper(), {}).get("description", ""),
                "quantity": qty,
                "unit_price_gbp": price,
                "line_total_gbp": _gbp(cents),
            }
            for code, qty, price, cents in lines
        ]
        order["lines_returned"] = len(lines)
        return {"found": True, "order": order}

    def list_orders(
        self,
        customer_id: str | None = None,
        country: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_total_gbp: float | None = None,
        sort: str = "date_desc",
        limit: int = 10,
    ) -> dict[str, Any]:
        def bad(msg: str) -> dict[str, Any]:
            return {"count": 0, "orders": [], "error": msg}

        limit = _clamp(limit, 1, 50, 10)
        if status is not None and status not in ORDER_STATUSES:
            return bad(f"status must be one of {list(ORDER_STATUSES)}")
        if sort not in ORDER_SORTS:
            return bad(f"sort must be one of {list(ORDER_SORTS)}")
        start = end = None
        if date_from is not None:
            start = _day(date_from)
            if start is None:
                return bad("date_from must be a date like 2011-10-01")
        if date_to is not None:
            end = _day(date_to)
            if end is None:
                return bad("date_to must be a date like 2011-10-31")
        floor = None
        if min_total_gbp is not None:
            try:
                floor = round(float(min_total_gbp) * 100)
            except (TypeError, ValueError):
                return bad("min_total_gbp must be a number")
        cid = norm_id(customer_id) if customer_id is not None else None
        want_country = str(country).strip().lower() if country else None

        rows = []
        for o in self._orders.values():
            day = o["invoice_date"][:10]
            if cid is not None and o["customer_id"] != cid:
                continue
            if want_country and o["country"].lower() != want_country:
                continue
            if status is not None and o["status"] != status:
                continue
            if start and day < start.isoformat():
                continue
            if end and day > end.isoformat():
                continue
            if floor is not None and o["total_cents"] < floor:
                continue
            rows.append(o)
        if sort == "date_desc":
            rows.sort(key=lambda o: (o["invoice_date"], o["id"]), reverse=True)
        elif sort == "date_asc":
            rows.sort(key=lambda o: (o["invoice_date"], o["id"]))
        elif sort == "total_desc":
            rows.sort(key=lambda o: (-o["total_cents"], o["id"]))
        else:
            rows.sort(key=lambda o: (o["total_cents"], o["id"]))
        by_country = Counter(o["country"] for o in rows)
        out: dict[str, Any] = {
            "count": len(rows),
            "total_gbp": _gbp(sum(o["total_cents"] for o in rows)),
            "counts_by_country": dict(sorted(by_country.items(), key=lambda kv: (-kv[1], kv[0]))),
            "counts_by_status": dict(sorted(Counter(o["status"] for o in rows).items())),
            "orders": [self._order_summary(o) for o in rows[:limit]],
        }
        if want_country and not rows and want_country not in {c.lower() for c in self._countries}:
            out["known_countries"] = self._countries
        return out

    def get_product(self, query: str, limit: int = 5) -> dict[str, Any]:
        limit = _clamp(limit, 1, 20, 5)
        text = str(query).strip()
        if not text:
            return {"found": False, "query": text, "count": 0, "products": []}
        exact = self._products.get(text.upper())
        if exact is not None:
            return {"found": True, "query": text, "count": 1, "products": [self._product_view(exact)]}
        words = text.lower().split()
        hits = [p for p in self._products.values()
                if all(w in p["description"].lower() for w in words)]
        hits.sort(key=lambda p: (-p["units_sold_in_window"], p["stock_code"]))
        return {
            "found": bool(hits),
            "query": text,
            "count": len(hits),
            "products": [self._product_view(p) for p in hits[:limit]],
        }

    def create_ticket(self, customer_id: str, subject: str, body: str) -> dict[str, Any]:
        cid = norm_id(customer_id)
        if cid not in self._customers:
            return {"created": False, "error": "unknown customer", "customer_id": cid}
        if not subject.strip() or len(subject) > MAX_SUBJECT_CHARS:
            return {"created": False, "error": f"subject must be 1-{MAX_SUBJECT_CHARS} characters"}
        if len(body) > MAX_BODY_CHARS:
            return {"created": False, "error": f"body must be at most {MAX_BODY_CHARS} characters"}
        ticket = Ticket(
            id=f"T-{9001 + len(self._tickets)}",
            customer_id=cid,
            subject=subject,
            body=body,
            created_at=self._clock().isoformat(timespec="seconds"),
        )
        self._tickets.append(ticket)
        if self._tickets_path is not None:
            self._tickets_path.write_text(
                json.dumps([t.model_dump() for t in self._tickets], indent=2) + "\n"
            )
        return {"created": True, "ticket": ticket.model_dump()}
