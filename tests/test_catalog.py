import pathlib

import pytest

from ops_agent.catalog import build_registry
from ops_agent.store import Store

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
NAMES = {
    "search_customers", "get_customer", "get_order",
    "list_orders", "check_inventory", "create_ticket",
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
    assert [s.name for s in reg.specs() if s.write] == ["create_ticket"]


@pytest.mark.parametrize("name,args", [
    ("search_customers", {"query": "ava"}),
    ("get_customer", {"customer_id": "C-1001"}),
    ("get_order", {"order_id": "O-5001"}),
    ("list_orders", {"status": "delivered", "limit": 5}),
    ("list_orders", {}),
    ("check_inventory", {"sku": "LOB-TNT-2P"}),
    ("create_ticket", {"customer_id": "C-1001", "subject": "s", "body": "b", "confirm": True}),
])
def test_each_tool_ok(reg, name, args):
    r = reg.call(name, args)
    assert r["ok"] is True and isinstance(r["data"], dict)


def test_miss_is_ok_not_error(reg):
    r = reg.call("get_order", {"order_id": "O-0"})
    assert r == {"ok": True, "data": {"found": False, "order_id": "O-0"}}


def test_unknown_tool(reg):
    assert reg.call("drop_tables", {})["error"]["code"] == "unknown_tool"


def test_create_ticket_needs_confirm(reg, tmp_path):
    args = {"customer_id": "C-1001", "subject": "s", "body": "b"}
    assert reg.call("create_ticket", args)["error"]["code"] == "confirmation_required"
    assert reg.call("create_ticket", {**args, "confirm": False})["error"]["code"] == "confirmation_required"
    assert not (tmp_path / "tickets.json").exists()


def test_bad_arguments(reg):
    assert reg.call("get_order", {})["error"]["code"] == "bad_arguments"
    assert reg.call("get_order", {"order_id": "O-1", "extra": 1})["error"]["code"] == "bad_arguments"
    assert reg.call("get_order", ["x"])["error"]["code"] == "bad_arguments"
