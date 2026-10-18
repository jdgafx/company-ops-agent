import json
import pathlib
import shutil
from collections import Counter

import pytest

from ops_agent.store import MAX_BODY_CHARS, MAX_SUBJECT_CHARS, Store

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def store():
    return Store(DATA)


@pytest.fixture(scope="module")
def raw_orders():
    return json.loads((DATA / "orders.json").read_text())


def test_counts_and_window(store):
    assert store.counts() == {
        "customers": 400, "orders": 3089, "products": 2774, "order_lines": 61962,
    }
    info = store.dataset_info()
    assert info["window"] == {"from": "2011-09-01", "to": "2011-11-30"}
    assert info["currency"] == "GBP" and info["doi"] == "10.24432/C5CG6D"


def test_search_by_country_and_unknown_country(store):
    r = store.search_customers(country="eire")
    assert r["count"] == 2 and {c["country"] for c in r["customers"]} == {"EIRE"}
    miss = store.search_customers(country="Ireland")
    assert miss["count"] == 0 and "EIRE" in miss["known_countries"]


def test_search_sort_query_and_limit(store):
    top = store.search_customers(sort_by="orders_desc", limit=2)
    assert top["count"] == 400 and top["customers"][0]["customer_id"] == "14911"
    assert top["customers"][0]["orders_in_window"] >= top["customers"][1]["orders_in_window"]
    assert len(store.search_customers(limit=1000)["customers"]) == 50
    assert len(store.search_customers(limit=0)["customers"]) == 1
    one = store.search_customers(query="12748")
    assert one["count"] == 1 and one["customers"][0]["country"] == "United Kingdom"
    assert "error" in store.search_customers(sort_by="nope")


def test_customers_carry_only_id_country_and_derived_figures(store):
    keys = set(store.search_customers()["customers"][0])
    assert keys == {"customer_id", "country", "orders_in_window",
                    "cancelled_orders_in_window", "invoiced_total_gbp_in_window"}


def test_get_customer(store):
    r = store.get_customer("12748")
    assert r["found"] and r["customer"]["country"] == "United Kingdom"
    assert r["order_count"] == 80 and r["orders_listed"] == 25 and len(r["orders"]) == 25
    assert r["orders"][0]["invoice_date"] >= r["orders"][-1]["invoice_date"]
    assert store.get_customer(12748)["customer"]["customer_id"] == "12748"  # int from the model
    assert store.get_customer("12748.0")["found"]
    assert store.get_customer("99999") == {"found": False, "customer_id": "99999"}


def test_get_order_totals_and_truncation(store, raw_orders):
    big = max(raw_orders, key=lambda o: len(o["lines"]))
    r = store.get_order(big["id"], max_lines=3)["order"]
    assert r["line_count"] == len(big["lines"]) and r["lines_returned"] == 3
    assert len(r["lines"]) == 3 and r["lines"][0]["description"]
    full = store.get_order(big["id"], max_lines=100)["order"]
    assert full["total_gbp"] == r["total_gbp"]
    assert store.get_order(big["id"], max_lines=0)["order"]["lines_returned"] == 1  # clamped


def test_get_order_cancelled_and_missing(store):
    o = store.get_order("c565085")["order"]  # lower-case is normalised
    assert o["status"] == "cancelled" and o["total_gbp"] == -17.9
    assert o["lines"][0]["quantity"] == -2 and o["lines"][0]["line_total_gbp"] == -17.9
    assert store.get_order("999999") == {"found": False, "order_id": "999999"}


def test_list_orders_filters_and_aggregates(store):
    everything = store.list_orders(limit=1000)
    assert everything["count"] == 3089 and len(everything["orders"]) == 50
    assert everything["counts_by_status"] == {"cancelled": 481, "invoiced": 2608}
    assert everything["counts_by_country"]["EIRE"] == 135
    eire = store.list_orders(country="EIRE", status="invoiced")
    assert eire["count"] == 106 and set(eire["counts_by_country"]) == {"EIRE"}
    assert all(o["status"] == "invoiced" for o in eire["orders"])
    mine = store.list_orders(customer_id="12395", status="invoiced", sort="date_asc")
    assert [o["order_id"] for o in mine["orders"]] == ["569496", "575151", "575154", "577551"]
    day = store.list_orders(date_from="2011-09-01", date_to="2011-09-01")
    assert day["count"] > 0 and all(o["invoice_date"].startswith("2011-09-01") for o in day["orders"])
    big = store.list_orders(min_total_gbp=2000, sort="total_desc")
    assert big["orders"][0]["total_gbp"] >= big["orders"][-1]["total_gbp"] >= 2000
    assert "lines" not in everything["orders"][0]


def test_list_orders_bad_inputs_are_structured_errors(store):
    for kwargs in ({"status": "shipped"}, {"sort": "x"}, {"date_from": "yesterday"},
                   {"date_to": "2011-13-01"}, {"min_total_gbp": "lots"}):
        r = store.list_orders(**kwargs)
        assert r["count"] == 0 and r["orders"] == [] and "error" in r
    nope = store.list_orders(country="Ireland")
    assert nope["count"] == 0 and "EIRE" in nope["known_countries"]


def test_get_product_by_code_and_words(store):
    p = store.get_product("85123a")["products"][0]
    assert p["stock_code"] == "85123A" and p["unit_price_gbp"] == 2.95
    assert p["units_sold_in_window"] == 5455 and p["orders_in_window"] == 208
    assert p["last_sold"] == "2011-11-30"
    by_words = store.get_product("white hanging heart")
    assert by_words["found"] and by_words["products"][0]["stock_code"] == "85123A"
    assert "on_hand" not in p  # the dataset has no stock levels
    assert store.get_product("NOPE-123")["found"] is False
    assert store.get_product("  ")["found"] is False


def test_product_figures_match_raw_invoices(store, raw_orders):
    units, orders = Counter(), Counter()
    for o in raw_orders:
        if o["status"] != "invoiced":
            continue
        for code, qty, _price in o["lines"]:
            units[code] += qty
        for code in {ln[0] for ln in o["lines"]}:
            orders[code] += 1
    for code in ("85123A", "22423", "10125"):
        p = store.get_product(code)["products"][0]
        assert p["units_sold_in_window"] == units[code] and p["orders_in_window"] == orders[code]


def test_create_ticket_success_persists(tmp_path):
    path = tmp_path / "tickets.json"
    s = Store(DATA, tickets_path=path)
    r = s.create_ticket("12748", "Late parcel", "Where is it?")
    assert r["created"] and r["ticket"]["id"] == "T-9001"
    saved = json.loads(path.read_text())
    assert len(saved) == 1 and saved[0]["subject"] == "Late parcel"
    second = Store(DATA, tickets_path=path).create_ticket(12395, "x", "y")
    assert second["ticket"]["id"] == "T-9002" and second["ticket"]["customer_id"] == "12395"


def test_create_ticket_unknown_customer(tmp_path):
    path = tmp_path / "tickets.json"
    r = Store(DATA, tickets_path=path).create_ticket("99999", "s", "b")
    assert r["created"] is False
    assert not path.exists()


def test_create_ticket_never_writes_data_dir():
    before = (DATA / "tickets.json").read_text()
    Store(DATA).create_ticket("12748", "s", "b")
    assert (DATA / "tickets.json").read_text() == before


def test_data_consistency(store, raw_orders):
    window = store.dataset_info()["window"]
    customers = json.loads((DATA / "customers.json").read_text())
    assert all(set(c) == {"id", "country"} for c in customers)  # no personal data exists
    ids = {c["id"]: c["country"] for c in customers}
    codes = {p["stock_code"] for p in json.loads((DATA / "products.json").read_text())}
    for o in raw_orders:
        assert ids[o["customer_id"]] == o["country"]
        assert window["from"] <= o["date"][:10] <= window["to"]
        assert (o["status"] == "cancelled") == o["id"].startswith("C")
        assert all(code in codes for code, _q, _p in o["lines"])
    for o in store._orders.values():
        assert (o["total_cents"] < 0) == (o["status"] == "cancelled")
    assert Counter(o["status"] for o in raw_orders)["invoiced"] == 2608


def test_missing_data_files_load_empty(tmp_path):
    s = Store(tmp_path)
    assert s.counts()["orders"] == 0 and s.examples() == []
    assert s.search_customers()["count"] == 0


def test_create_ticket_rejects_oversized_text(tmp_path):
    shutil.copytree(DATA, tmp_path / "d")
    store = Store(tmp_path / "d", tickets_path=tmp_path / "t.json")
    too_long_subject = store.create_ticket("12748", "s" * (MAX_SUBJECT_CHARS + 1), "b")
    assert too_long_subject["created"] is False
    too_long_body = store.create_ticket("12748", "ok", "b" * (MAX_BODY_CHARS + 1))
    assert too_long_body["created"] is False
    assert not (tmp_path / "t.json").exists()  # nothing persisted on rejection
