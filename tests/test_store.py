import json
import pathlib

import pytest

from ops_agent.store import Store

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def store():
    return Store(DATA)


def test_search_hit_by_name_and_email(store):
    assert store.search_customers("ava alvarez")["count"] == 1
    assert store.search_customers("AVA.ALVAREZ@EXAMPLE")["count"] == 1


def test_search_miss_and_limit(store):
    assert store.search_customers("zzz-nobody")["count"] == 0
    r = store.search_customers("example.com", limit=1000)
    assert r["count"] == 30 and len(r["customers"]) == 30
    assert len(store.search_customers("example.com", limit=0)["customers"]) == 1


def test_get_customer(store):
    r = store.get_customer("C-1001")
    assert r["found"] and r["customer"]["id"] == "C-1001" and r["order_ids"]
    assert store.get_customer("C-9999") == {"found": False, "customer_id": "C-9999"}


def test_customers_without_orders(store):
    assert store.get_customer("C-1030")["order_ids"] == []


def test_get_order(store):
    r = store.get_order("O-5001")
    assert r["found"]
    o = r["order"]
    assert o["total_usd"] == round(sum(i["qty"] * i["unit_price_usd"] for i in o["items"]), 2)
    assert store.get_order("O-0")["found"] is False


def test_list_orders(store):
    everything = store.list_orders(limit=50)
    assert everything["count"] == 60 and len(everything["orders"]) == 50
    shipped = store.list_orders(status="shipped")
    assert shipped["orders"] and all(o["status"] == "shipped" for o in shipped["orders"])
    bad = store.list_orders(status="nonsense")
    assert bad["count"] == 0 and bad["orders"] == [] and "error" in bad


def test_check_inventory(store):
    hit = store.check_inventory("LOB-TNT-2P")
    assert hit["found"] and hit["in_stock"] is True
    out = store.check_inventory("LOB-TNT-4P")
    assert out["found"] and out["in_stock"] is False
    assert store.check_inventory("NOPE")["found"] is False


def test_create_ticket_success_persists(tmp_path):
    path = tmp_path / "tickets.json"
    s = Store(DATA, tickets_path=path)
    r = s.create_ticket("C-1001", "Late parcel", "Where is it?")
    assert r["created"] and r["ticket"]["id"] == "T-9001"
    saved = json.loads(path.read_text())
    assert len(saved) == 1 and saved[0]["subject"] == "Late parcel"
    assert Store(DATA, tickets_path=path).create_ticket("C-1002", "x", "y")["ticket"]["id"] == "T-9002"


def test_create_ticket_unknown_customer(tmp_path):
    path = tmp_path / "tickets.json"
    r = Store(DATA, tickets_path=path).create_ticket("C-0", "s", "b")
    assert r["created"] is False
    assert not path.exists()


def test_create_ticket_never_writes_data_dir():
    before = (DATA / "tickets.json").read_text()
    Store(DATA).create_ticket("C-1001", "s", "b")
    assert (DATA / "tickets.json").read_text() == before


def test_data_consistency(store):
    for o in store._orders.values():
        assert o.customer_id in store._customers
        assert o.total_usd == round(sum(i.qty * i.unit_price_usd for i in o.items), 2)
        assert all(i.sku in store._inventory for i in o.items)
    assert sum(1 for i in store._inventory.values() if i.on_hand == 0) >= 2


def test_create_ticket_rejects_oversized_text(tmp_path):
    import shutil

    from ops_agent.store import MAX_BODY_CHARS, MAX_SUBJECT_CHARS, Store
    shutil.copytree("data", tmp_path / "d")
    store = Store(tmp_path / "d", tickets_path=tmp_path / "t.json")
    too_long_subject = store.create_ticket("C-1001", "s" * (MAX_SUBJECT_CHARS + 1), "b")
    assert too_long_subject["created"] is False
    too_long_body = store.create_ticket("C-1001", "ok", "b" * (MAX_BODY_CHARS + 1))
    assert too_long_body["created"] is False
    assert not (tmp_path / "t.json").exists()  # nothing persisted on rejection
