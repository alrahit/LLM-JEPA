"""spider_exec.py - execution-accuracy scoring on the local Spider sqlite files (stdlib only)."""
import collections
import os
import pathlib
import re
import sqlite3
import time

DB_PAT = re.compile(r"For db_id:\[(.+?)\]")
ORDER_PAT = re.compile(r"\border\s+by\b", re.I)


def extract_db_id(user_text):
    m = DB_PAT.search(user_text)
    assert m, "no db_id in prompt"
    return m.group(1)


def run_sql(db_path, sql, timeout_s=10.0):
    """Returns (True, rows) or (False, error_text). Read-only connection with a wall-clock timeout."""
    conn = sqlite3.connect(pathlib.Path(db_path).as_uri() + "?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.time() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100000)
    try:
        return True, conn.execute(sql).fetchall()
    except Exception as e:  # syntax error, missing column, timeout, several statements, ...
        return False, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def norm_rows(rows):
    return [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows]


def strict_equal(pred_rows, gold_rows, gold_sql):
    """Same rows in the same column order; row order only matters if the gold query has ORDER BY."""
    p, g = norm_rows(pred_rows), norm_rows(gold_rows)
    if ORDER_PAT.search(gold_sql):
        return p == g
    return collections.Counter(p) == collections.Counter(g)


def loose_equal(pred_rows, gold_rows):
    """Like strict, but ignores row order and column order (secondary metric)."""
    key = lambda r: tuple(sorted(repr(v) for v in r))
    return collections.Counter(map(key, norm_rows(pred_rows))) == collections.Counter(map(key, norm_rows(gold_rows)))


def norm_sql(sql):
    return " ".join(sql.strip().rstrip(";").lower().split())


def score(db_dir, db_id, gold_sql, pred_sql):
    path = os.path.join(db_dir, db_id, db_id + ".sqlite")
    g_ok, g = run_sql(path, gold_sql)
    if pred_sql.strip():
        p_ok, p = run_sql(path, pred_sql)
    else:
        p_ok, p = False, "empty prediction"
    out = {"gold_ok": g_ok, "pred_ok": p_ok, "gold_empty": bool(g_ok and len(g) == 0),
           "strict": False, "loose": False, "exact_str": norm_sql(pred_sql) == norm_sql(gold_sql)}
    if g_ok and p_ok:
        out["strict"] = strict_equal(p, g, gold_sql)
        out["loose"] = loose_equal(p, g)
    if not p_ok:
        out["pred_error"] = p
    return out


def summarize(items):
    n = len(items)
    nonempty = [x for x in items if x["gold_ok"] and not x["gold_empty"]]
    return {
        "n": n,
        "gold_errors": sum(not x["gold_ok"] for x in items),
        "gold_empty": sum(x["gold_empty"] for x in items),
        "pred_errors": sum(not x["pred_ok"] for x in items),
        "exec_acc_all": sum(x["strict"] for x in items) / max(1, n),
        "n_nonempty": len(nonempty),
        "exec_acc_nonempty": sum(x["strict"] for x in nonempty) / max(1, len(nonempty)),
        "exec_acc_loose_all": sum(x["loose"] for x in items) / max(1, n),
        "string_exact_all": sum(x["exact_str"] for x in items) / max(1, n),
    }
