import pathlib

import pytest

from ops_agent.catalog import build_registry
from ops_agent.store import Store
from ops_agent.tools import summarize_result

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
NAMES = {
    "search_customers", "get_customer", "get_order",
    "list_orders", "get_product", "create_ticket",
}


@pytest.fixture
def reg(tmp_path):
    return build_registry(Store(DATA, tickets_path=tmp_path / "tickets.json"))


def test_tool_names_and_schemas(reg):
    assert {s.name for s in reg.specs()} == NAMES
    for s in reg.specs():
        assert s.description
        assert s.input_schema["type"] == "object"
        assert set(s.input_schema["required"]) <= set(s.input_schema["properties"])
        assert s.summarize is not None
    assert [s.name for s in reg.specs() if s.write] == ["create_ticket"]


def test_descriptions_state_currency_and_dataset_limits(reg):
    text = " ".join(s.description for s in reg.specs())
    assert "GBP" in text and "NO stock levels" in text and "EIRE" in text
    assert "inventory" not in text.lower()


@pytest.mark.parametrize("name,args", [
    ("search_customers", {"country": "Germany"}),
    ("search_customers", {}),
    ("get_customer", {"customer_id": "12748"}),
    ("get_order", {"order_id": "565083"}),
    ("list_orders", {"status": "invoiced", "limit": 5}),
    ("list_orders", {}),
    ("get_product", {"query": "85123A"}),
    ("create_ticket", {"customer_id": "12748", "subject": "s", "body": "b", "confirm": True}),
])
def test_each_tool_ok(reg, name, args):
    r = reg.call(name, args)
    assert r["ok"] is True and isinstance(r["data"], dict)


def test_miss_is_ok_not_error(reg):
    r = reg.call("get_order", {"order_id": "1"})
    assert r == {"ok": True, "data": {"found": False, "order_id": "1"}}


def test_unknown_tool(reg):
    assert reg.call("drop_tables", {})["error"]["code"] == "unknown_tool"
    assert reg.call("check_inventory", {"sku": "x"})["error"]["code"] == "unknown_tool"


def test_create_ticket_needs_confirm(reg, tmp_path):
    args = {"customer_id": "12748", "subject": "s", "body": "b"}
    assert reg.call("create_ticket", args)["error"]["code"] == "confirmation_required"
    assert reg.call("create_ticket", {**args, "confirm": False})["error"]["code"] == "confirmation_required"
    assert not (tmp_path / "tickets.json").exists()


def test_bad_arguments(reg):
    assert reg.call("get_order", {})["error"]["code"] == "bad_arguments"
    assert reg.call("get_order", {"order_id": "1", "extra": 1})["error"]["code"] == "bad_arguments"
    assert reg.call("get_order", ["x"])["error"]["code"] == "bad_arguments"
    assert reg.call("get_product", {})["error"]["code"] == "bad_arguments"


def summary(reg, name, args):
    return summarize_result(reg.get(name), args, reg.call(name, args))


def test_summaries_are_short_and_specific(reg):
    assert summary(reg, "get_customer", {"customer_id": "12748"}) == (
        "customer 12748 (United Kingdom), 80 orders")
    assert summary(reg, "get_customer", {"customer_id": "99999"}) == "customer 99999 not found"
    assert summary(reg, "get_order", {"order_id": "C565085"}) == (
        "order C565085 (cancelled), 1 lines, GBP -17.90")
    assert summary(reg, "list_orders", {"country": "EIRE"}).startswith("135 orders, total GBP")
    assert summary(reg, "get_product", {"query": "85123A"}) == (
        "product 85123A, 5455 units sold in window")
    assert summary(reg, "get_product", {"query": "zzz"}) == "no product matching 'zzz'"
    assert summary(reg, "search_customers", {"country": "Germany"}).endswith("in Germany")
    assert "created for customer 12748" in summary(
        reg, "create_ticket", {"customer_id": "12748", "subject": "s", "body": "b", "confirm": True})
    assert summary(reg, "get_order", {}) == "error bad_arguments"
