"""
Runner de experimentos MiniZinc (model-aware)
---------------------------------------------

- Inyecta una lista de variables de decisión según el tipo de modelo
  (Sudoku / Reunión / Kakuro / Secuencia / Acertijo / Rectángulo),
  y sustituye la línea 'solve ... satisfy;' por una de tres estrategias de búsqueda
  claramente diferenciadas para análisis comparativo:
    1) ff_min       -> first_fail + indomain_min     (completa)
    2) wdeg_split   -> dom_w_deg   + indomain_split  (completa)
    3) inorder_min  -> input_order + indomain_min    (completa)

- Compila a FlatZinc y ejecuta con estadísticas.
- Guarda un CSV de resultados y, opcionalmente, una "shortlist" de corridas interesantes.
- Comentarios normalizados en español y limpieza de temporales.

Notas:
* Se evita referenciar símbolos inexistentes (p. ej., G) en la inyección.
* Si la compilación o ejecución con la estrategia falla por identificadores indefinidos,
  se reintenta con el modelo original (sin inyección de solve).
* Nuevo (ACERTIJO): si el modelo es 'acertijo' no requiere archivos .dzn. Se ejecuta el
  modelo “tal cual”, cambiando únicamente la heurística/solver.
"""

import argparse
import subprocess
import tempfile
import re
import csv
import os
import sys
import glob
import shutil
import math
from collections import defaultdict

# ==============================================================

# Localiza la primera línea 'solve ... satisfy;'
SOLVE_REGEX = re.compile(
    r"(?m)^[ \t]*(?!%)[ \t]*solve\s*(::[^\n]*)?\s*satisfy\s*;",
    re.IGNORECASE
)

# Etiqueta explícita en cabecera (opcional):
#   % MODEL_ID: sudoku
#   % MODEL_ID: reunion
#   % MODEL_ID: kakuro
#   % MODEL_ID: secuencia
#   % MODEL_ID: acertijo
#   % MODEL_ID: rectangulo
MODEL_ID_RE = re.compile(r"^\s*%+\s*MODEL_ID:\s*(\w+)", re.IGNORECASE | re.MULTILINE)

# Heurísticas para detectar tipo de modelo (no exhaustivas)
HINTS = {
    "sudoku": {
        "must": [re.compile(r"\barray\s*\[\s*S\s*,\s*S\s*\]\s*of\s*var\b.*:\s*X\b", re.DOTALL)],
        "any":  [re.compile(r"\bG\s*;\s*$", re.MULTILINE),
                 re.compile(r"\ball_different\(\s*\[\s*X\[r,c\]")],
    },
    "reunion": {
        "must": [re.compile(r"\barray\s*\[\s*S\s*\]\s*of\s*var\s*POS\s*:\s*POS_OF\b"),
                 re.compile(r"\barray\s*\[\s*POS\s*\]\s*of\s*var\s*S\s*:\s*PER_AT\b"),
                 re.compile(r"\binverse\s*\(\s*POS_OF\s*,\s*PER_AT\s*\)")],
        "any":  [re.compile(r"\bNEXT\b"), re.compile(r"\bSEP\b"), re.compile(r"\bDIST\b")],
    },
    # Kakuro (grilla x[Cols,Rows], máscara white[Cols,Rows]) — ahora usa SEARCH_VARS
    "kakuro": {
        "must": [
            re.compile(r"\barray\s*\[[^\]]+\]\s*of\s*var\b.*:\s*x\b", re.IGNORECASE | re.DOTALL),
            re.compile(r"\barray\s*\[[^\]]+\]\s*of\s*int\b.*:\s*white\b", re.IGNORECASE | re.DOTALL),
        ],
        "any": [re.compile(r"\bCols\b"), re.compile(r"\bRows\b"), re.compile(r"\bint_lin_\w+\b")],
    },
    # Secuencia: vector 1D de decisión llamado x
    "secuencia": {
        "must": [re.compile(r"\barray\s*\[[^\]]+\]\s*of\s*var\b.*:\s*x\b", re.IGNORECASE | re.DOTALL)],
        "any":  [re.compile(r"\bx\["), re.compile(r"\ball_different|\bint_lin_")],
    },
    # Acertijo: vector 1D de decisión llamado vars
    "acertijo": {
        "must": [re.compile(r"\barray\s*\[[^\]]+\]\s*of\s*var\b.*:\s*vars\b", re.IGNORECASE | re.DOTALL)],
        "any":  [re.compile(r"\bvars\["), re.compile(r"\ball_different|\bint_lin_")],
    },
    # Rectángulo: el modelo expone un vector de búsqueda llamado SEARCH_VARS
    "rectangulo": {
        "must": [re.compile(r"\barray\s*\[[^\]]+\]\s*of\s*var\b.*:\s*SEARCH_VARS\b", re.IGNORECASE | re.DOTALL)],
        "any":  [re.compile(r"\bSEARCH_VARS\[")],
    },
}

# Señales de lista de branching ya definida (case-insensitive)
def pick_existing_branch_name(model_text: str) -> str | None:
    """
    Si el modelo define una lista de branching conocida, devuelve su nombre
    tal como aparece en el modelo (preserva capitalización).
    Busca SEARCH_VARS, DECISION_VARS y BRANCH_VARS (case-insensitive).
    """
    for label in ("SEARCH_VARS", "DECISION_VARS", "BRANCH_VARS"):
        m = re.search(rf"\b({label})\b", model_text, re.IGNORECASE)
        if m:
            return model_text[m.start():m.end()]
    return None

def detect_model_kind(model_text: str) -> str:
    """Devuelve 'sudoku' | 'reunion' | 'kakuro' | 'secuencia' | 'acertijo' | 'rectangulo' | 'unknown'."""
    m = MODEL_ID_RE.search(model_text)
    if m:
        tag = m.group(1).strip().lower()
        if tag in ("sudoku", "reunion", "kakuro", "secuencia", "acertijo", "rectangulo"):
            return tag
    for kind, patt in HINTS.items():
        if all(rx.search(model_text) for rx in patt["must"]) and any(rx.search(model_text) for rx in patt["any"]):
            return kind
    return "unknown"

def ensure_branch_vars(model_text: str, kind: str) -> tuple[str, str | None]:
    """
    Garantiza una lista 1D de variables de ramificación:
    - Reusa SEARCH_VARS / DECISION_VARS / BRANCH_VARS si existen.
    - Sudoku: X[r,c] (filtra por G[r,c]==0 si G existe) -> DECISION_VARS.
    - Reunión: POS_OF[p] -> DECISION_VARS.
    - Kakuro: x[c,r] en celdas white -> SEARCH_VARS (cambio pedido).
    - Secuencia: x[i] -> DECISION_VARS.
    - Acertijo: vars[i] -> DECISION_VARS.
    - Rectángulo: si existe SEARCH_VARS declarado, se usa tal cual (no se inyecta otra lista).
    - Inserta justo antes de la primera línea 'solve' cuando corresponda.
    """
    # Si ya existe alguna lista conocida, reusar su nombre exacto
    name = pick_existing_branch_name(model_text)
    if name:
        return model_text, name

    # Localizar primera 'solve'
    m = SOLVE_REGEX.search(model_text)
    if not m:
        return model_text, None

    snippet = None

    if kind == "sudoku":
        has_G = bool(re.search(r"^\s*(?:array\s*\[.*\]\s*of\s*)?int\s*:\s*G\b", model_text, re.MULTILINE))
        if has_G:
            snippet = (
                "% === Inyectado por runner: DECISION_VARS (Sudoku, con G) ===\n"
                "array[int] of var int: DECISION_VARS =\n"
                "  [ X[r,c] |\n"
                "    r in index_set_1_of_2(X),\n"
                "    c in index_set_2_of_2(X)\n"
                "    where G[r,c] = 0\n"
                "  ];\n"
            )
        else:
            snippet = (
                "% === Inyectado por runner: DECISION_VARS (Sudoku) ===\n"
                "array[int] of var int: DECISION_VARS =\n"
                "  [ X[r,c] |\n"
                "    r in index_set_1_of_2(X),\n"
                "    c in index_set_2_of_2(X)\n"
                "  ];\n"
            )
        name = "DECISION_VARS"

    elif kind == "reunion":
        snippet = (
            "% === Inyectado por runner: DECISION_VARS (Reunión) ===\n"
            "array[int] of var int: DECISION_VARS = [ POS_OF[p] | p in S ];\n"
        )
        name = "DECISION_VARS"

    elif kind == "kakuro":
        # Cambio solicitado: usar SEARCH_VARS
        snippet = (
            "% === Inyectado por runner: SEARCH_VARS (Kakuro) ===\n"
            "array[int] of var 0..9: SEARCH_VARS =\n"
            "  [ x[c,r] |\n"
            "    c in Cols,\n"
            "    r in Rows\n"
            "    where white[c,r] = 1\n"
            "  ];\n"
        )
        name = "SEARCH_VARS"

    elif kind == "secuencia":
        snippet = (
            "% === Inyectado por runner: DECISION_VARS (Secuencia) ===\n"
            "array[int] of var int: DECISION_VARS =\n"
            "  [ x[i] |\n"
            "    i in index_set_1_of_1(x)\n"
            "  ];\n"
        )
        name = "DECISION_VARS"

    elif kind == "acertijo":
        snippet = (
            "% === Inyectado por runner: DECISION_VARS (Acertijo) ===\n"
            "array[int] of var int: DECISION_VARS =\n"
            "  [ vars[i] |\n"
            "    i in index_set_1_of_1(vars)\n"
            "  ];\n"
        )
        name = "DECISION_VARS"

    elif kind == "rectangulo":
        # No inyectamos nada: se asume que el modelo define SEARCH_VARS.
        has_search = bool(re.search(r"\barray\s*\[[^\]]+\]\s*of\s*var\b.*:\s*SEARCH_VARS\b",
                                    model_text, re.IGNORECASE | re.DOTALL))
        if has_search:
            return model_text, "SEARCH_VARS"
        # Si no existe, no arriesgamos a inventar; devolvemos sin cambios.
        return model_text, None

    else:
        # Heurístico Kakuro-like → ahora también en SEARCH_VARS
        has_x = bool(re.search(r"\barray\s*\[[^\]]+\]\s*of\s*var\b.*:\s*x\b", model_text, re.IGNORECASE | re.DOTALL))
        has_white = bool(re.search(r"\barray\s*\[[^\]]+\]\s*of\s*int\b.*:\s*white\b", model_text, re.IGNORECASE | re.DOTALL))
        if has_x and has_white:
            snippet = (
                "% === Inyectado por runner: SEARCH_VARS (Kakuro-like) ===\n"
                "array[int] of var 0..9: SEARCH_VARS =\n"
                "  [ x[c,r] |\n"
                "    c in Cols,\n"
                "    r in Rows\n"
                "    where white[c,r] = 1\n"
                "  ];\n"
            )
            name = "SEARCH_VARS"
        else:
            return model_text, None

    # Insertar justo antes de la primera línea 'solve'
    i = m.start()
    patched = model_text[:i] + snippet + model_text[i:]
    return patched, name

# ==============================================================

# Tres estrategias diferenciadas (sin 'default'):
_SOLVE_LINE = "solve :: int_search({VARS}, {VARH}, {VALH}, complete) satisfy;"
SOLVE_TEMPLATES_BY_MODEL = {
    kind: {
        "ff_min":      _SOLVE_LINE.format(VARS="{VARS}", VARH="first_fail",  VALH="indomain_min"),
        "wdeg_split":  _SOLVE_LINE.format(VARS="{VARS}", VARH="dom_w_deg",   VALH="indomain_split"),
        "inorder_min": _SOLVE_LINE.format(VARS="{VARS}", VARH="input_order", VALH="indomain_min"),
        "inorder_split": _SOLVE_LINE.format(VARS="{VARS}", VARH="input_order", VALH="indomain_split"),
    }
    for kind in ("sudoku", "reunion", "kakuro", "secuencia", "acertijo", "rectangulo", "unknown")
}

STRAT_ALIASES = {
    "domdeg_split": "wdeg_split",
}

def normalize_strategy_name(s: str) -> str:
    s = s.strip().lower()
    return STRAT_ALIASES.get(s, s)

def inject_solve_by_kind(model_text: str, strategy: str, kind: str) -> str:
    """
    Sustituye la línea 'solve ... satisfy;' según la estrategia pedida.
    - Asegura la lista de branching adecuada (SEARCH_VARS/DECISION_VARS según el modelo).
    - Si no puede asegurar variables de ramificación, devuelve el texto original.
    """
    strategy = normalize_strategy_name(strategy)
    templates = SOLVE_TEMPLATES_BY_MODEL.get(kind, SOLVE_TEMPLATES_BY_MODEL["unknown"])
    if strategy not in templates:
        return model_text

    # Asegurar la lista de branching (intenta inyectar si no existe)
    txt, varname = ensure_branch_vars(model_text, kind)
    if varname is None:
        # No podemos inyectar; devolvemos el texto original
        return model_text

    if not SOLVE_REGEX.search(txt):
        return model_text

    solve_line = templates[strategy].format(VARS=varname)
    return SOLVE_REGEX.sub(solve_line, txt, count=1)

# ==============================================================

def format_time_sci(t, digits=3):
    if t is None:
        return None
    return f"{t:.{digits}e}"

def parse_stats(mzn_text: str):
    """
    Extrae '%%%mzn-stat:' y '%%%mzn-status' de stdout+stderr.
    Devuelve dict con casts básicos y compatibilidad 'fail' -> 'failures'.
    """
    stats = {}
    for line in mzn_text.splitlines():
        if line.startswith("%%%mzn-stat:") or line.startswith("%%%mzn-status"):
            kv = line.split(":", 1)[1].strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                stats[k.strip()] = v.strip()
            else:
                stats["status"] = kv.strip()

    def cast_float(k):
        if k in stats:
            try:
                stats[k] = float(stats[k])
            except:
                pass

    def cast_int(k):
        if k in stats:
            try:
                stats[k] = int(float(stats[k]))
            except:
                pass

    for k in ["time", "initTime", "solveTime"]:
        cast_float(k)
    for k in ["nodes", "failures", "solutions", "restarts", "peakDepth", "variables", "constraints", "fail", "nSolutions"]:
        cast_int(k)
    if "failures" not in stats and "fail" in stats:
        stats["failures"] = stats["fail"]
    if "solutions" not in stats and "nSolutions" in stats:
        stats["solutions"] = stats["nSolutions"]

    return stats

def compute_total_time(stats):
    """
    Normaliza el tiempo total:
    - Usa 'time' si existe; si no, 'initTime' + 'solveTime'; si no, None.
    """
    t = stats.get("time")
    if isinstance(t, (int, float)):
        return t
    it = stats.get("initTime")
    st = stats.get("solveTime")
    if isinstance(it, (int, float)) or isinstance(st, (int, float)):
        return (it or 0.0) + (st or 0.0)
    return None

def pareto_min(rows, keys):
    """Índices de la frontera de Pareto minimizando en 'keys'."""
    idxs = []
    for i, ri in enumerate(rows):
        if any(ri.get(k) is None for k in keys):
            continue
        dominated = False
        for j, rj in enumerate(rows):
            if i == j:
                continue
            if any(rj.get(k) is None for k in keys):
                continue
            better_or_equal_all = all(rj[k] <= ri[k] for k in keys)
            strictly_better_one = any(rj[k] < ri[k] for k in keys)
            if better_or_equal_all and strictly_better_one:
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs

def iqr_fences(values, k=1.5):
    """Cercas IQR (Tukey) para detectar outliers; devuelve (lo, hi) o (None, None)."""
    if len(values) < 4:
        return (None, None)
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    lower = vs[:mid]
    upper = vs[mid + 1:] if n % 2 == 1 else vs[mid:]
    def median(a):
        m = len(a) // 2
        return (a[m] + a[~m]) / 2 if len(a) % 2 == 0 else a[m]
    q1 = median(lower) if lower else None
    q3 = median(upper) if upper else None
    if q1 is None or q3 is None:
        return (None, None)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)

def shortlist_from_rows(rows, topk=2, delta=1.5, iqr_k=1.5):
    """
    Construye shortlist por archivo:
    - Mejores/peores tiempos
    - Frontera de Pareto (time_raw, nodes, failures)
    - Outliers por IQR en time_raw y nodes
    - Gaps entre solvers para una misma estrategia
    - Anomalías (rc != 0 o solutions != 1)
    """
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)

    shortlisted = []

    for f, group in by_file.items():
        g_time = [r for r in group if r.get("time_raw") is not None]
        if g_time:
            best = sorted(
                g_time,
                key=lambda r: (r.get("time_raw", float("inf")),
                               r.get("nodes") if r.get("nodes") is not None else math.inf)
            )[:topk]
            worst = sorted(
                g_time,
                key=lambda r: (-(r.get("time_raw", -float("inf"))),
                               -(r.get("nodes") if r.get("nodes") is not None else -1))
            )[:topk]
            for r in best:
                shortlisted.append((r, "best-time"))
            for r in worst:
                shortlisted.append((r, "worst-time"))

        idxs = pareto_min(group, keys=["time_raw", "nodes", "failures"])
        for i in idxs:
            shortlisted.append((group[i], "pareto-front"))

        for metric in ["time_raw", "nodes"]:
            vals = [r[metric] for r in group if r.get(metric) is not None]
            lo, hi = iqr_fences(vals, k=iqr_k)
            if lo is None:
                continue
            for r in group:
                v = r.get(metric)
                if v is None:
                    continue
                if v < lo or v > hi:
                    shortlisted.append((r, f"outlier-{metric}"))

        for r in group:
            if (r.get("rc") not in (0, None)) or (r.get("solutions") not in (1, None)):
                shortlisted.append((r, "anomaly"))

        by_strat = defaultdict(list)
        for r in group:
            by_strat[r["strategy"]].append(r)
        for strat, g in by_strat.items():
            times = [r["time_raw"] for r in g if r.get("time_raw") is not None]
            nodes = [r["nodes"] for r in g if r.get("nodes") is not None]

            def ratio(vs):
                return max(vs) / min(vs) if len(vs) >= 2 and min(vs) > 0 else 1.0

            if times and ratio(times) >= delta:
                tmin = min(g, key=lambda r: r.get("time_raw", float("inf")))
                tmax = max(g, key=lambda r: r.get("time_raw", -1.0))
                shortlisted.append((tmin, f"solver-gap-time({ratio(times):.2f}x)"))
                shortlisted.append((tmax, f"solver-gap-time({ratio(times):.2f}x)"))

            if nodes and ratio(nodes) >= delta:
                nmin = min(g, key=lambda r: r.get("nodes", float("inf")))
                nmax = max(g, key=lambda r: r.get("nodes", -1))
                shortlisted.append((nmin, f"solver-gap-nodes({ratio(nodes):.2f}x)"))
                shortlisted.append((nmax, f"solver-gap-nodes({ratio(nodes):.2f}x)"))

    # De-duplicación por (file, solver, strategy)
    seen = set()
    out_rows = []
    for r, reason in shortlisted:
        key = (r["file"], r["solver"], r["strategy"])
        if key in seen:
            continue
        seen.add(key)
        o = dict(r)
        o["reason"] = reason
        out_rows.append(o)
    return out_rows

def emit_latex_table(rows, path, caption="Shortlist de corridas interesantes", label="tab:shortlist"):
    """Emite tabla LaTeX compacta con la shortlist."""
    cols = ["file", "solver", "strategy", "reason", "time", "nodes", "failures", "peakDepth", "solutions", "status"]
    header = ["Archivo", "Solver", "Estrategia", "Motivo", "Tiempo (s)", "Nodes", "Failures", "Depth", "Sol.", "Status"]
    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("  \\centering")
    lines.append("  \\small")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{tabular}{l l l l r r r r r l}")
    lines.append("    \\hline")
    lines.append("    " + " & ".join([f'\\textbf{{{h}}}' for h in header]) + " \\\\")
    lines.append("    \\hline")
    for r in rows:
        vals = [r.get(k, "") for k in cols]
        try:
            if isinstance(vals[4], float):
                vals[4] = f"{vals[4]:.3f}"
        except Exception:
            pass
        line = " & ".join([str(v) if v is not None else "" for v in vals]) + " \\\\"
        lines.append("    " + line)
    lines.append("    \\hline")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def audit_redundancy_in_fzn(fzn_path: str) -> dict:
    """
    Auditoría robusta del FlatZinc:
      - allDiff_count: cuenta apariciones de all_different (_int) si se mantiene global.
      - sumX_count: cuenta int_lin_eq con TODOS los coeficientes = 1 y TODAS las vars empezando por 'X'
      - rhs_examples: algunos RHS observados.
    """
    try:
        with open(fzn_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return {"allDiff_count": None, "sumX_count": None, "redundancy_on": None, "rhs_examples": []}

    allDiff_count = len(re.findall(r'\ball[_ ]?different(?:_int)?\b', txt))
    pat = re.compile(
        r'constraint\s+int_lin_eq\s*\(\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*,\s*(-?\d+)\s*\)',
        re.DOTALL
    )

    def _all_ones(coeffs_str: str) -> bool:
        items = [c.strip() for c in coeffs_str.split(",") if c.strip()]
        return bool(items) and all(c == "1" for c in items)

    VAR_TOKEN = re.compile(r'^[A-Za-z_][A-Za-z0-9_\[\], ]*$')
    def _all_X_vars(vars_str: str) -> bool:
        items = [v.strip() for v in vars_str.split(",") if v.strip()]
        if not items:
            return False
        for v in items:
            if not VAR_TOKEN.match(v):
                return False
            base = v.split("[", 1)[0].split(",", 1)[0].split(" ", 1)[0].split("_", 1)[0]
            if not base.startswith("X"):
                return False
        return True

    sumX_count = 0
    rhs_examples = []
    for m in pat.finditer(txt):
        coeffs_str, vars_str, rhs = m.group(1), m.group(2), m.group(3)
        if _all_ones(coeffs_str) and _all_X_vars(vars_str):
            sumX_count += 1
            if len(rhs_examples) < 5:
                rhs_examples.append(int(rhs))

    redundancy_on = (sumX_count >= 5)

    return {
        "allDiff_count": allDiff_count,
        "sumX_count": sumX_count,
        "rhs_examples": rhs_examples,
        "redundancy_on": redundancy_on,
    }

# ==============================================================

def main():
    ap = argparse.ArgumentParser(description="Runner de experimentos para modelos MiniZinc (con inyección de estrategias).")
    # Rutas base
    ap.add_argument("--base-dir", required=True, help="Directorio base para salidas")
    ap.add_argument("--model", required=True, help="Ruta al modelo .mzn (relativa a base-dir si no es absoluta)")
    ap.add_argument("--data-dir", required=True, help="Directorio con .dzn (relativa a base-dir si no es absoluta)")
    # Solvers y estrategias
    ap.add_argument("--solver", nargs="+", default=["gecode", "chuffed"], help="Lista de solvers (p. ej., gecode chuffed)")
    ap.add_argument("--strategy", nargs="+", default=["ff_min", "wdeg_split", "inorder_min", "inorder_split"],
                    help="Estrategias: ff_min | wdeg_split (alias: domdeg_split) | inorder_min")
    ap.add_argument("--time-limit", type=int, default=60000, help="Límite de tiempo en ms para minizinc")
    # Enumerar todas las soluciones (opcional)
    ap.add_argument("--all-solutions", action="store_true", default=False,
                    help="Enumerar todas las soluciones (no detener en la primera)")
    # Salidas
    ap.add_argument("--out", default="results/results.csv", help="CSV de resultados (relativa a base-dir)")
    ap.add_argument("--shortlist", action="store_true", default=False, help="Generar shortlist y artefactos")
    ap.add_argument("--shortlist-out", default="results/shortlist.csv", help="CSV de shortlist (relativa a base-dir)")
    ap.add_argument("--topk", type=int, default=2, help="Top-K de mejores/peores por instancia")
    ap.add_argument("--delta", type=float, default=1.5, help="Umbral de gap entre solvers (ratio)")
    ap.add_argument("--iqr-k", type=float, default=1.5, help="K de IQR para outliers")
    ap.add_argument("--copy-shortlist-to", default="results/shortlist_artifacts",
                    help="Carpeta (relativa a base-dir) para copiar .log/.sol de la shortlist")
    args = ap.parse_args()

    # Helper para normalizar rutas relativas a base-dir
    def rel(p):
        return p if os.path.isabs(p) else os.path.join(args.base_dir, p)

    # Cargar modelo
    model_path = rel(args.model)
    with open(model_path, "r", encoding="utf-8") as f:
        model_text = f.read()
    model_kind = detect_model_kind(model_text)

    # Enumerar instancias .dzn (para 'acertijo' son opcionales)
    data_glob_dir = rel(args.data_dir)
    data_files = sorted(glob.glob(os.path.join(data_glob_dir, "*.dzn")))
    if not data_files:
        if model_kind == "acertijo":
            # Ejecutamos sin .dzn (solo cambia solver/heurística)
            data_files = [None]
            print("[INFO] Modelo 'acertijo' sin .dzn: se ejecutará únicamente el .mzn (sin datos).")
        else:
            print("No se encontraron archivos .dzn en", data_glob_dir, file=sys.stderr)
            sys.exit(2)

    # Directorios de salida
    out_csv_path = rel(args.out)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    runs_dir = rel("results/artifacts")
    os.makedirs(runs_dir, exist_ok=True)
    fzn_dir = rel("results/flat")
    os.makedirs(fzn_dir, exist_ok=True)

    # Normalizar estrategias
    norm_strategies = []
    for s in args.strategy:
        ns = normalize_strategy_name(s)
        if ns not in ("ff_min", "wdeg_split", "inorder_min", "indom-split", "inorder_split"):
            print(f"[WARN] estrategia '{s}' no soportada; se ignora.", file=sys.stderr)
            continue
        norm_strategies.append(ns)
    if not norm_strategies:
        norm_strategies = ["ff_min", "wdeg_split", "inorder_min"]

    rows = []

    for solver in args.solver:
        for strat in norm_strategies:
            # Generar versión del modelo para la estrategia (si aplica)
            try:
                mod_txt = inject_solve_by_kind(model_text, strat, model_kind)
            except Exception as e:
                print(f"[WARN] {solver}/{strat}: error inyectando estrategia ({e}); se usa modelo original.", file=sys.stderr)
                mod_txt = model_text

            # Vista previa de la línea solve
            try:
                msolve = SOLVE_REGEX.search(mod_txt)
                solve_preview = mod_txt[msolve.start():msolve.end()] if msolve else "<sin-solve>"
                print(f"[DEBUG] {solver}/{strat}: solve => {solve_preview}")
            except Exception:
                pass

            # Archivo temporal del modelo inyectado
            tmpm_path = None
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False, encoding="utf-8") as tmpm:
                    tmpm.write(mod_txt)
                    tmpm.flush()
                    tmpm_path = tmpm.name

                # (Opcional) Conteo de lista de branching: prueba con primera instancia (o sin dzn si acertijo)
                if re.search(r"\b(SEARCH_VARS|DECISION_VARS|BRANCH_VARS)\b", mod_txt, re.IGNORECASE):
                    tmpc_path = None
                    try:
                        with tempfile.NamedTemporaryFile("w", suffix=".mzn", delete=False, encoding="utf-8") as tmpc:
                            tmpc.write(mod_txt + "\n")
                            tmpc.write(
                                'output ["#BRANCH_LIST=", '
                                'show(if exists(SEARCH_VARS) then length(SEARCH_VARS) '
                                'else if exists(DECISION_VARS) then length(DECISION_VARS) else 0 endif), "\\n"];'
                            )
                            tmpc.flush()
                            tmpc_path = tmpc.name

                        df0 = data_files[0]
                        cmd_preview = ["minizinc", "--solver", solver, "--time-limit", "1000", tmpc_path]
                        if df0:
                            cmd_preview.append(df0)
                        preview_proc = subprocess.run(cmd_preview, capture_output=True, text=True)
                        if preview_proc.returncode == 0:
                            mcount = re.search(r"#BRANCH_LIST=\s*(\d+)", preview_proc.stdout)
                            if mcount:
                                print(f"[DEBUG] {solver}/{strat}: BRANCH_LIST size = {mcount.group(1)}")
                    except Exception:
                        pass
                    finally:
                        if tmpc_path:
                            try: os.unlink(tmpc_path)
                            except Exception: pass

                # Compilar y ejecutar cada instancia (o sin instancia si acertijo)
                for data_path in data_files:
                    if data_path:
                        base = os.path.splitext(os.path.basename(data_path))[0]
                    else:
                        base = os.path.splitext(os.path.basename(model_path))[0] + "__nodzn"

                    tag  = f"{base}__{solver}__{strat}"

                    # Compilación a FlatZinc
                    fzn_path = os.path.join(fzn_dir, f"{tag}.fzn")
                    cmd_compile = ["minizinc", "--solver", solver, "-c", tmpm_path]
                    if data_path:
                        cmd_compile.append(data_path)
                    cmd_compile.extend(["-o", fzn_path])

                    proc_c = subprocess.run(cmd_compile, capture_output=True, text=True)
                    stdout_c, stderr_c, rc_c = proc_c.stdout, proc_c.stderr, proc_c.returncode

                    # Reintento con modelo original si hay identificadores indefinidos
                    if rc_c != 0 and ("undefined identifier" in (stdout_c + stderr_c).lower()):
                        print(f"[WARN] {solver}/{strat}: recompilo con modelo original (identificador indefinido).", file=sys.stderr)
                        fzn_path = os.path.join(fzn_dir, f"{tag}.orig.fzn")
                        cmd_compile2 = ["minizinc", "--solver", solver, "-c", model_path]
                        if data_path:
                            cmd_compile2.append(data_path)
                        cmd_compile2.extend(["-o", fzn_path])
                        proc_c2 = subprocess.run(cmd_compile2, capture_output=True, text=True)
                        stdout_c, stderr_c, rc_c = proc_c2.stdout, proc_c2.stderr, proc_c2.returncode

                    # Inspección de líneas 'branch' en FlatZinc
                    if os.path.exists(fzn_path):
                        try:
                            with open(fzn_path, "r", encoding="utf-8", errors="ignore") as ff:
                                branch_lines = [ln.strip() for ln in ff if ln.startswith("branch")]
                            if branch_lines:
                                print(f"[DEBUG] branch-lines ({solver}/{strat}/{base}):")
                                for ln in branch_lines[:8]:
                                    print("   ", ln)
                        except Exception:
                            pass

                    # Si no hay .fzn, omitir ejecución
                    if not os.path.exists(fzn_path):
                        print(f"[WARN] {solver}/{strat}/{base}: .fzn no generado; se omite.", file=sys.stderr)
                        rows.append({
                            "file": base, "solver": solver, "strategy": strat,
                            "rc": rc_c, "status": None, "time_raw": None, "time": None,
                            "nodes": None, "failures": None, "peakDepth": None, "solutions": None,
                            "restarts": None, "initTime": None, "solveTime": None
                        })
                        continue

                    # Ejecución con estadísticas
                    cmd = ["minizinc", "--solver", solver, "--statistics", "--time-limit", str(args.time_limit), fzn_path]
                    if args.all_solutions:
                        cmd.insert(1, "--all-solutions")

                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode

                    # Guardar artefactos de salida
                    runs_dir = rel("results/artifacts")
                    with open(os.path.join(runs_dir, f"{tag}.sol.txt"), "w", encoding="utf-8") as fsol:
                        fsol.write(stdout)
                    with open(os.path.join(runs_dir, f"{tag}.log.txt"), "w", encoding="utf-8") as flog:
                        flog.write("CMD: " + " ".join(cmd) + "\n\n")
                        flog.write(stdout + "\n\n--- STDERR ---\n" + stderr)

                    # Parseo de estadísticas y normalización de tiempo total
                    stats = parse_stats(stdout + "\n" + stderr)
                    tsec_raw = compute_total_time(stats)
                    tsec = format_time_sci(tsec_raw, digits=3)

                    rows.append({
                        "file": base,
                        "solver": solver,
                        "strategy": strat,
                        "rc": rc,
                        "status": stats.get("status"),
                        "time_raw": tsec_raw,
                        "time": tsec,
                        "nodes": stats.get("nodes"),
                        "failures": stats.get("failures"),
                        "peakDepth": stats.get("peakDepth"),
                        "solutions": stats.get("solutions"),
                        "restarts": stats.get("restarts"),
                        "initTime": stats.get("initTime"),
                        "solveTime": stats.get("solveTime"),
                    })
                    print(f"[{solver}/{strat}] {base}: rc={rc}, status={stats.get('status')}, "
                          f"time={tsec}, nodes={stats.get('nodes')}, failures={stats.get('failures')}")

            finally:
                if tmpm_path:
                    try: os.unlink(tmpm_path)
                    except Exception: pass

    # CSV de resultados
    with open(out_csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=["file","solver","strategy","rc","status","time_raw","time",
                        "nodes","failures","peakDepth","solutions","restarts","initTime","solveTime"]
        )
        w.writeheader()
        w.writerows(rows)
    print("Saved results CSV:", out_csv_path)

    # Shortlist opcional
    if args.shortlist:
        sl = shortlist_from_rows(rows, topk=args.topk, delta=args.delta, iqr_k=args.iqr_k)

        sl_csv = rel(args.shortlist_out)
        os.makedirs(os.path.dirname(sl_csv) or ".", exist_ok=True)
        with open(sl_csv, "w", newline="", encoding="utf-8") as fsl:
            w = csv.DictWriter(
                fsl,
                fieldnames=["file","solver","strategy","reason","time","nodes","failures",
                            "peakDepth","solutions","restarts","rc","status","initTime","solveTime","time_raw"]
            )
            w.writeheader()
            for r in sl:
                w.writerow(r)
        print(f"Saved shortlist CSV: {sl_csv} ({len(sl)} rows)")

        # Copia de artefactos de la shortlist
        target_dir = rel(args.copy_shortlist_to)
        os.makedirs(target_dir, exist_ok=True)
        copied = 0
        for r in sl:
            tag = f"{r['file']}__{r['solver']}__{r['strategy']}"
            for ext in [".log.txt", ".sol.txt"]:
                src = os.path.join(rel("results"), "artifacts", f"{tag}{ext}")
                if os.path.exists(src):
                    dst = os.path.join(target_dir, f"{tag}{ext}")
                    shutil.copy2(src, dst)
                    copied += 1
        print(f"Copied {copied} artifacts to {target_dir}")

if __name__ == "__main__":
    main()
