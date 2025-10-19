% --------------------------------------------------------------
% jobshop_mantenimiento.mzn — JSSP con ventanas de mantenimiento por máquina
% - Objetivo: minimizar el makespan (end).
% - Salida: end y matriz de tiempos de inicio s[i,j].
% --------------------------------------------------------------

% ----------------------- Parámetros y conjuntos ----------------
int: jobs;                                    % número de trabajos
set of int: JOB = 1..jobs;

int: tasks;                                   % número de operaciones por trabajo
set of int: TASK = 1..tasks;

array[JOB, TASK] of int: proc_time;           % tiempos de proceso p_ij >= 0

% Horizonte trivial por suma de duraciones (cota superior)
int: total = sum(i in JOB, j in TASK)(proc_time[i,j]);

% Dígitos para salida alineada
int: digs = ceil(log(10.0, max(1,total)));    % evitar log(0)

% ------------- Ventanas de mantenimiento por máquina -----------
% Para modelar "varios intervalos" por máquina, usamos un máximo y un flag de uso.
int: MAX_MAINT_WINDOWS;                                       % máximo de ventanas por máquina
array[TASK, 1..MAX_MAINT_WINDOWS] of int: maint_start;        % inicio de mantenimiento en máquina m, ventana k
array[TASK, 1..MAX_MAINT_WINDOWS] of int: maint_end;          % fin de mantenimiento en máquina m, ventana k
array[TASK, 1..MAX_MAINT_WINDOWS] of bool: maint_active;      % true si la ventana (m,k) es válida

% Recomendación de validez cuando maint_active[m,k] = true:
% 0 <= maint_start[m,k] < maint_end[m,k] <= total

% ----------------------- Variables -----------------------------
array[JOB, TASK] of var 0..total: s;           % tiempos de inicio s[i,j]
var 0..total: end;                             % makespan

% ----------------------- Predicados ----------------------------
predicate no_overlap(var int: s1, int: d1, var int: s2, int: d2) =
  s1 + d1 <= s2 \/ s2 + d2 <= s1;

% ----------------------- Restricciones -------------------------
% (1) Secuencia dentro de cada trabajo
constraint
  forall(i in JOB)(
    forall(j in 1..tasks-1)( s[i,j] + proc_time[i,j] <= s[i,j+1] ) /\
    s[i,tasks] + proc_time[i,tasks] <= end
  );

% (2) No solape en cada máquina (tarea j es la máquina j)
constraint
  forall(m in TASK)(
    forall(i,k in JOB where i < k)(
      no_overlap(s[i,m], proc_time[i,m], s[k,m], proc_time[k,m])
    )
  );

% (3) Mantenimientos: bloquear ventanas [maint_start[m,k], maint_end[m,k]) en máquina m
%     Se modela como "no solape" entre cada operación (i,m) y el intervalo de mantenimiento.
constraint
  forall(m in TASK, k in 1..MAX_MAINT_WINDOWS where maint_active[m,k]) (
    let { int: dur = maint_end[m,k] - maint_start[m,k] } in
      (dur > 0) /\
      forall(i in JOB)(
        no_overlap(s[i,m], proc_time[i,m], maint_start[m,k], dur)
      )
  );

% ----------------------- Redundantes -------------------
% Cotas inferiores del makespan:
%  - Por trabajo: end >= suma de sus duraciones (camino crítico trivial).
%  - Por máquina: end >= carga total en la máquina.
constraint
  forall(i in JOB)(
    end >= sum(j in TASK)(proc_time[i,j])
  );

constraint
  forall(m in TASK)(
    end >= sum(i in JOB)(proc_time[i,m])
  );

% ----------------------- Búsqueda -------------------------------
% Variables de ramificación: todos los s[i,j]
array[int] of var int: BRANCH_VARS = [ s[i,j] | i in JOB, j in TASK ];

% [H1] Heurística 1 — clásica
% solve :: int_search(BRANCH_VARS, first_fail, indomain_min, complete) minimize end;

% [H2] Heurística 2 — informada por conflictos
% solve :: int_search(BRANCH_VARS, dom_w_deg, indomain_split, complete) minimize end;

% [H3] Heurística 3 — orden de entrada
% solve :: int_search(BRANCH_VARS, input_order, indomain_min, complete) minimize end;

% [D] Default del solver
solve minimize end;

% ----------------------- Salida --------------------------------
output
[
  "Resumen del cronograma\n",
  "-----------------------\n",
  "Makespan (end) = ", show(end), "\n\n",

  "Tabla de inicios y fines por Job (s..f)\n",
  " (s..f indica [inicio..fin) en cada máquina)\n\n",

  "          "  % cabeceras de máquinas
] 
++
[ 
  " M" ++ show(j) ++ "        " | j in TASK 
]
++
[
  "\n"
]
++
[
  % una fila por Job con los 3 (o tasks) pares s..f
  "Job " ++ show(i) ++ "  | " ++
  concat([
    let {
      int: s_ij = fix(s[i,j]),
      int: f_ij = s_ij + proc_time[i,j]
    } in
      % ancho fijo: s..f con relleno
      show_int(2, s_ij) ++ ".." ++ show_int(2, f_ij) ++ "   "
    | j in TASK
  ]) ++ "\n"
  | i in JOB
]
++
[
  "\nMantenimientos por máquina [start,end):\n",
  "---------------------------------------\n"
]
++
[
  "M" ++ show(m) ++ ": " ++
  concat([
    if maint_active[m,k]
    then "[" ++ show(maint_start[m,k]) ++ "," ++ show(maint_end[m,k]) ++ ") "
    else ""
    endif
    | k in 1..MAX_MAINT_WINDOWS
  ]) ++ "\n"
  | m in TASK
];

