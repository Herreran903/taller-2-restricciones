% MODEL_ID: jobshop
% --------------------------------------------------------------
% JOBSHOP_MANTENIMIENTO.MZN — JSSP con ventanas de mantenimiento por máquina
% - Objetivo: minimizar el makespan (END).
% - Salida: END y matriz de tiempos de inicio S[I,J].
% --------------------------------------------------------------

% ----------------------- Parámetros y conjuntos ----------------
int: JOBS;
set of int: JOB = 1..JOBS;

int: TASKS;
set of int: TASK = 1..TASKS;

array[JOB, TASK] of int: PROC_TIME;

int: MAX_MAINT_WINDOWS;
array[TASK, 1..MAX_MAINT_WINDOWS] of int: MAINT_START;
array[TASK, 1..MAX_MAINT_WINDOWS] of int: MAINT_END;
array[TASK, 1..MAX_MAINT_WINDOWS] of bool: MAINT_ACTIVE;

int: TOTAL_PROC  = sum(i in JOB, j in TASK)(PROC_TIME[i,j]);
int: TOTAL_MAINT = sum(m in TASK, k in 1..MAX_MAINT_WINDOWS where MAINT_ACTIVE[m,k])
                   (max(0, MAINT_END[m,k] - MAINT_START[m,k]));
int: H = TOTAL_PROC + TOTAL_MAINT;

% ----------------------- Variables -----------------------------
array[JOB, TASK] of var 0..H: S;
var 0..H: END;

% ----------------------- Predicados ----------------------------
predicate NO_OVERLAP(var int: S1, int: D1, var int: S2, int: D2) =
  S1 + D1 <= S2 \/ S2 + D2 <= S1;

% ----------------------- Restricciones -------------------------
% (1) Secuencia dentro de cada trabajo.
constraint
  forall(i in JOB)(
    forall(j in 1..TASKS-1)( S[i,j] + PROC_TIME[i,j] <= S[i,j+1] ) /\
    S[i,TASKS] + PROC_TIME[i,TASKS] <= END
  );

% (2) No solape en cada máquina.
constraint
  forall(m in TASK)(
    forall(i,k in JOB where i < k)(
      NO_OVERLAP(S[i,m], PROC_TIME[i,m], S[k,m], PROC_TIME[k,m])
    )
  );

% (3) Mantenimientos: bloquea (MAINT_START[m,k], MAINT_END[m,k]) en la máquina m.
constraint
  forall(m in TASK, k in 1..MAX_MAINT_WINDOWS where MAINT_ACTIVE[m,k]) (
    let { int: DUR = MAINT_END[m,k] - MAINT_START[m,k] } in
      (DUR > 0) /\
      forall(i in JOB)(
        NO_OVERLAP(S[i,m], PROC_TIME[i,m], MAINT_START[m,k], DUR)
      )
  );

% ----------------------- Redundantes ---------------------------
% (4) Cotas inferiores del makespan por trabajo.
constraint
  forall(i in JOB)(
    END >= sum(j in TASK)(PROC_TIME[i,j])
  );

% (5) Cotas inferiores del makespan por maquina.
constraint
  forall(m in TASK)(
    END >= sum(i in JOB)(PROC_TIME[i,m])
  );

% ----------------------- Simetria ---------------------------
% (6) Rompimiento de simetría entre jobs idénticos.
constraint
  forall(i,k in JOB where i < k /\
         forall(j in TASK)(PROC_TIME[i,j] = PROC_TIME[k,j])) (
    lex_lesseq([ S[i,j] | j in TASK ],
               [ S[k,j] | j in TASK ])
  );

% ----------------------- Búsqueda -------------------------------
% Variables de ramificación: todos los S[i,j].
array[int] of var int: BRANCH_VARS = [ S[i,j] | i in JOB, j in TASK ];

% [H1] Heurística 1 — clásica
% solve :: int_search(BRANCH_VARS, first_fail, indomain_min, complete) minimize END;

% [H2] Heurística 2 — informada por conflictos
% solve :: int_search(BRANCH_VARS, dom_w_deg, indomain_split, complete) minimize END;

% [H3] Heurística 3 — orden de entrada
% solve :: int_search(BRANCH_VARS, input_order, indomain_min, complete) minimize END;

% [D] Default del solver
solve minimize END;

% ----------------------- Salida -------------------------
function int: DIGITS(int: n) =
  if n < 10 then 1 else 1 + DIGITS(n div 10) endif;

output
let { int: W = DIGITS(H) } in
[
  "Makespan = ", show(END), "\n\n"
]
++
[
  concat([
    "Job ", show(i), ": ",
    concat([
      let {
        int: S_IJ = fix(S[i,j]),
        int: F_IJ = S_IJ + PROC_TIME[i,j]
      } in
        show_int(W, S_IJ) ++ ".." ++ show_int(W, F_IJ) ++ "  "
      | j in TASK
    ]),
    "\n"
  ])
  | i in JOB
];
