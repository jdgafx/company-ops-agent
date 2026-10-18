"""Build the data/ subset from the UCI "Online Retail II" workbook (not a project dependency).

Run it in a throwaway venv that has pandas and openpyxl, with the workbook path as an argument:

    python -I scripts/build_dataset.py /path/to/online_retail_II.xlsx --out data

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii (Daqing Chen, CC BY 4.0,
DOI 10.24432/C5CG6D). The workbook is read as data only; nothing in it is executed.

Selection rule (deterministic, no randomness):
1. Keep rows with InvoiceDate from 2011-09-01 to 2011-11-30 and a non-empty Customer ID.
2. Drop customers whose rows in the window carry more than one Country value.
3. Drop whole non-cancellation invoices that contain a line with quantity <= 0 or price <= 0
   (stock adjustments and write-offs, not sales).
4. Rank the remaining customers by number of distinct sales invoices in the window, most first,
   ties by customer id ascending. Keep the first TOP_N.
5. Keep every sales invoice of those customers (status "invoiced") and every cancellation
   invoice of those customers (invoice number starts with "C", status "cancelled").
No value is invented. Customers carry only the dataset's Customer ID and Country.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter
from decimal import Decimal

WINDOW_FROM = "2011-09-01"
WINDOW_TO = "2011-11-30"
TOP_N = 400


def money(value: float) -> float:
    """Prices as the dataset prints them: at most 4 decimals, no float noise."""
    return float(Decimal(repr(round(float(value), 4))))


def build(xlsx: pathlib.Path, top_n: int) -> dict:
    import pandas as pd  # build-time only

    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df[df["Customer ID"].notna()]
    stamp = df["InvoiceDate"].dt.strftime("%Y-%m-%d")
    df = df[(stamp >= WINDOW_FROM) & (stamp <= WINDOW_TO)].copy()
    df["customer"] = df["Customer ID"].astype("int64").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["cancelled"] = df["Invoice"].str.startswith("C")

    multi = df.groupby("customer")["Country"].nunique()
    df = df[df["customer"].isin(multi[multi == 1].index)]

    sales = df[~df["cancelled"]]
    bad = sales[(sales["Quantity"] <= 0) | (sales["Price"] <= 0)]["Invoice"].unique()
    df = df[~df["Invoice"].isin(bad)]
    sales = df[~df["cancelled"]]

    counts = sales.groupby("customer")["Invoice"].nunique().reset_index(name="n")
    counts = counts.sort_values(["n", "customer"], ascending=[False, True], kind="stable")
    chosen = set(counts["customer"].head(top_n))
    df = df[df["customer"].isin(chosen)]

    customers = sorted(
        ({"id": c, "country": g["Country"].iloc[0]} for c, g in df.groupby("customer")),
        key=lambda r: int(r["id"]),
    )

    orders = []
    for inv, g in df.groupby("Invoice", sort=False):
        first = g.iloc[0]
        orders.append({
            "id": inv,
            "date": g["InvoiceDate"].min().strftime("%Y-%m-%d %H:%M"),
            "customer_id": first["customer"],
            "country": first["Country"],
            "status": "cancelled" if first["cancelled"] else "invoiced",
            "lines": [
                [r.StockCode, int(r.Quantity), money(r.Price)] for r in g.itertuples()
            ],
        })
    orders.sort(key=lambda o: (o["date"], o["id"]))

    # Product description: most frequent in the kept rows (ties: alphabetical). Unit price:
    # most frequent in kept sales rows, else kept cancellation rows (ties: lowest).
    def modal(values, key):
        ranked = Counter(values).most_common()
        top = ranked[0][1]
        return min((v for v, c in ranked if c == top), key=key)

    products = []
    for code, g in df.groupby("StockCode"):
        descs = [d for d in g["Description"] if d]
        priced = g[~g["cancelled"]] if (~g["cancelled"]).any() else g
        products.append({
            "stock_code": code,
            "description": modal(descs, str) if descs else "",
            "unit_price_gbp": modal([money(p) for p in priced["Price"]], float),
        })
    products.sort(key=lambda p: p["stock_code"])

    return {"customers": customers, "orders": orders, "products": products,
            "dropped_adjustment_invoices": len(bad)}


def write_lines(path: pathlib.Path, rows: list) -> None:
    """One JSON record per line inside a JSON array: valid JSON, small, diff-friendly."""
    body = ",\n".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) for r in rows)
    path.write_text("[\n" + body + "\n]\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("xlsx", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data"))
    ap.add_argument("--top-n", type=int, default=TOP_N)
    args = ap.parse_args()
    built = build(args.xlsx, args.top_n)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("customers", "orders", "products"):
        write_lines(args.out / f"{name}.json", built[name])
    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "name": "UCI Online Retail II, subset",
        "source_url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "license": "CC BY 4.0 (attribution: Daqing Chen)",
        "doi": "10.24432/C5CG6D",
        "date_pulled": "2026-10-18",
        "currency": "GBP",
        "window": {"from": WINDOW_FROM, "to": WINDOW_TO},
        "top_n_customers": args.top_n,
    })
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in built.items() if isinstance(v, list)},
          "dropped adjustment invoices:", built["dropped_adjustment_invoices"])


if __name__ == "__main__":
    main()
