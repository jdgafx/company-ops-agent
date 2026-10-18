# Data source and attribution

This folder holds a subset of the UCI Machine Learning Repository data set "Online Retail II".

- Dataset: Online Retail II, https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Donor and author: Daqing Chen. Licence: Creative Commons Attribution 4.0 (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- DOI: 10.24432/C5CG6D
- Pulled: 2026-10-18 from https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- Changes: filtered and reshaped by `scripts/build_dataset.py` (rule in its docstring and in the README). Derived figures (`*_in_window`, order totals) are computed at load time by `src/ops_agent/store.py`.

Files: `customers.json` (id, country), `orders.json` (invoice id, date, customer id, country, status, lines as `[stock_code, quantity, unit_price_gbp]`), `products.json` (stock code, description, usual unit price in GBP), `meta.json` (source facts, window, example questions), `tickets.json` (empty; tickets the service creates are not written here).

The data holds no personal data. Customers are only the dataset's numeric Customer ID and country.
