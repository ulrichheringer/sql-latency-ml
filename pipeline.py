"""SQL latency regression using transparent, lexical query features."""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = ["estimated_rows", "joins", "predicates", "group_by", "order_by", "has_limit",
            "limit_value", "subquery", "distinct", "aggregate_calls", "query_length"]
TABLES = ["users", "orders", "events", "products", "payments", "sessions", "tickets", "teams"]


def extract_features(sql: str, estimated_rows=None):
    text = str(sql)
    lower = text.lower()
    row_hint = re.search(r"(?:estimated_rows|rows)\s*=\s*(\d+)", lower)
    rows = int(estimated_rows) if estimated_rows is not None else int(row_hint.group(1)) if row_hint else 1000
    where = re.search(r"\bwhere\b(.*?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", lower, re.S)
    predicates = 0 if not where else 1 + len(re.findall(r"\b(?:and|or)\b", where.group(1)))
    limit = re.search(r"\blimit\s+(\d+)", lower)
    return {
        "estimated_rows": max(rows, 1),
        "joins": len(re.findall(r"\bjoin\b", lower)),
        "predicates": predicates,
        "group_by": int(bool(re.search(r"\bgroup\s+by\b", lower))),
        "order_by": int(bool(re.search(r"\border\s+by\b", lower))),
        "has_limit": int(limit is not None),
        "limit_value": int(limit.group(1)) if limit else 0,
        "subquery": int(len(re.findall(r"\bselect\b", lower)) > 1),
        "distinct": int(bool(re.search(r"\bdistinct\b", lower))),
        "aggregate_calls": len(re.findall(r"\b(?:count|sum|avg|min|max)\s*\(", lower)),
        "query_length": len(text),
    }


def _dataset(seed=SEED, rows=1800):
    rng = random.Random(seed)
    queries, latency = [], []
    for _ in range(rows):
        table = rng.choice(TABLES)
        joins = rng.randint(0, 5)
        used = {table}
        join_sql = []
        for _ in range(joins):
            other = rng.choice([t for t in TABLES if t not in used])
            used.add(other)
            join_sql.append(f"JOIN {other} ON {table}.id = {other}.{table[:-1]}_id")
        predicates = rng.randint(0, 5)
        where_sql = " WHERE " + " AND ".join(f"{table}.field{i} IS NOT NULL" for i in range(predicates)) if predicates else ""
        grouped = rng.random() < 0.35
        group_sql = f" GROUP BY {table}.category" if grouped else ""
        sorted_query = rng.random() < 0.4
        order_sql = f" ORDER BY {table}.created_at DESC" if sorted_query else ""
        limit = rng.choice([0, 50, 100, 500, 1000])
        limit_sql = f" LIMIT {limit}" if limit else ""
        subquery = rng.random() < 0.15
        subquery_prefix = " AND " if predicates else " WHERE "
        subquery_sql = f"{subquery_prefix}{table}.id IN (SELECT id FROM {rng.choice(TABLES)})" if subquery else ""
        estimated_rows = int(np.exp(rng.uniform(np.log(50), np.log(150000))))
        select = f"SELECT COUNT(*) FROM {table}" if grouped else f"SELECT {table}.* FROM {table}"
        sql = f"{select} {' '.join(join_sql)}{where_sql}{subquery_sql}{group_sql}{order_sql}{limit_sql} /* estimated_rows={estimated_rows} */"
        features = extract_features(sql)
        work = (0.004 * estimated_rows + 38 * joins + 21 * subquery + 11 * grouped
                + 7 * sorted_query + 2.2 * predicates - 0.012 * limit
                + rng.gauss(0, 18))
        queries.append(sql)
        latency.append(max(1.0, 8 + work))
    X = pd.DataFrame([extract_features(q) for q in queries], columns=FEATURES)
    return X, np.asarray(latency)


def _model(seed=SEED):
    return RandomForestRegressor(n_estimators=220, min_samples_leaf=2, max_features=0.9,
                                 random_state=seed, n_jobs=1)


def sample_input():
    return {"sql": "SELECT o.* FROM orders o JOIN users u ON o.user_id = u.id WHERE o.status = 'open' GROUP BY o.id ORDER BY o.created_at DESC LIMIT 100 /* estimated_rows=24000 */"}


def train_model(seed=SEED):
    X, y = _dataset(seed)
    return _model(seed).fit(X, y)


def predict(model, payload):
    if isinstance(payload, str):
        sql, rows = payload, None
    else:
        sql, rows = payload["sql"], payload.get("estimated_rows")
    X = pd.DataFrame([extract_features(sql, rows)], columns=FEATURES)
    return {"estimated_latency_ms": round(float(model.predict(X)[0]), 2), "features": X.iloc[0].to_dict()}


def evaluate(seed=SEED):
    X, y = _dataset(seed)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = _model(seed).fit(x_train, y_train)
    pred = model.predict(x_test)
    return {"mae_ms": round(float(mean_absolute_error(y_test, pred)), 4),
            "rmse_ms": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "r2": round(float(r2_score(y_test, pred)), 4), "test_rows": int(len(y_test))}
