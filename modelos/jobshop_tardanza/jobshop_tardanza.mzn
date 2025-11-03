% --------------------------------------------------------------
% JOBSHOP_TARDANZA.MZN — JSSP con ventanas de mantenimiento por máquina
% (estructura base; ventanas no incluidas aún)
% - Objetivo: minimizar tardanza ponderada (w = sum_j weights[j]*T[j])
% - Salida: w y matriz de tiempos de inicio s[i,j]
% --------------------------------------------------------------

% ======================= Identificación =======================
% MODEL_ID: jobshop

% ======================= Parámetros ===========================
int: jobs;                          % número de trabajos
set of int: JOB = 1..jobs;

int: tasks;                         % número de tareas por trabajo
set of int: TASK = 1..tasks;

array[JOB, TASK] of int: d;         % duraciones d[i,j]
array[JOB] of int: weights;         % pesos por trabajo (para tardanza)
array[JOB] of int: due_dates;       % due dates por trabajo

int: total = sum(i in JOB, j in TASK)(d[i,j]); % duración total máx
int: digs  = ceil(log(10.0, total));          % dígitos para salida

% ======================= Variables ============================
array[JOB, TASK] of var 0..total: s;          % start times
array[JOB] of var 0..total: end_job;          % fin de cada trabajo
array[JOB] of var 0..total: T;                % tardanza por trabajo
var 0..total: end;                            % makespan
var 0..sum(j in JOB)(weights[j])*total: w;    % tardanza ponderada total

% ======================= Predicados ===========================
predicate no_overlap(var int: s1, int: d1, var int: s2, int: d2) =
    s1 + d1 <= s2 \/ s2 + d2 <= s1;
    
predicate identical_jobs(int: i, int: k) =
  weights[i] = weights[k] /\
  due_dates[i] = due_dates[k] /\
  forall(j in TASK)( d[i,j] = d[k,j] );

% ======================= Restricciones ========================
% (1) Secuencia interna de tareas por trabajo
constraint
  forall (i in JOB) (
    forall (j in 1..tasks-1) (s[i,j] + d[i,j] <= s[i,j+1])
    /\ s[i,tasks] + d[i,tasks] <= end
  );

% (2) No solape en cada “máquina” implícita por índice de tarea
constraint
  forall (j in TASK) (
    forall (i, k in JOB where i < k) (
      no_overlap(s[i,j], d[i,j], s[k,j], d[k,j])
    )
  );

% (3) Fin de cada trabajo
constraint
  forall (j in JOB) (
    end_job[j] = s[j,tasks] + d[j,tasks]
  );

% (4) Definición de tardanza T[j] = max(0, end_job[j] - due_dates[j])
constraint
  forall (j in JOB) (
    T[j] >= end_job[j] - due_dates[j] /\ T[j] >= 0
  );

% (5) Función objetivo (tardanza ponderada)
constraint
  w = sum (j in JOB) (weights[j] * T[j]);

% % ======================= Redundantes ==========================
% % (Opcional) Redundantes típicos podrían incluir lower/upper bounds por máquina,
% % o acotar 'end' con la suma mínima de la “cadena crítica”, etc.
% % De momento se dejan fuera para mantener el modelo limpio.

% %(1) Ventanas de tiempo por operación (EST/LST)
% % Acota cada inicio s[i,j] entre: 
% % - EST[i,j]: lo más temprano posible (suma de duraciones previas del job i)
% % - LST[i,j]: lo más tarde posible para que aún quepa lo que falta del job i
% array[JOB,TASK] of int: EST = 
%   array2d(JOB, TASK, [ sum(t in 1..j-1)(d[i,t]) | i in JOB, j in TASK ]);

% array[JOB,TASK] of int: REM_AFTER =
%   array2d(JOB, TASK, [ sum(t in j..tasks)(d[i,t]) | i in JOB, j in TASK ]);

% array[JOB,TASK] of int: LST = 
%   array2d(JOB, TASK, [ total - REM_AFTER[i,j] | i in JOB, j in TASK ]);
  
% constraint forall(i in JOB, j in TASK)(
%   s[i,j] >= EST[i,j] /\ s[i,j] <= LST[i,j]
% );

% %(2) Cota por carga de “máquina” (una por índice de tarea) 
% % end no puede ser menor que la suma de duraciones asignadas a cada máquina j.

% array[TASK] of int: load_by_machine = 
%   [ sum(i in JOB)(d[i,j]) | j in TASK ];

% constraint forall(j in TASK)( end >= load_by_machine[j] );

% % (3) Cota por “trabajo más largo”
% % end tampoco puede ser menor que la duración total de cada job.

% array[JOB] of int: job_length = [ sum(j in TASK)(d[i,j]) | i in JOB ];

% constraint forall(i in JOB)( end >= job_length[i] );

% constraint forall(i in JOB)( end >= end_job[i] );
  
%  % ======================= Simetría =============================
%   % Rompe simetría entre trabajos idénticos con orden lexicográfico estricto.
%   % Para cada par (i,k) idéntico (i<k), existe un primer j donde difieren
%   % y allí forzamos s[i,j] < s[k,j], manteniendo igualdad en posiciones previa
%   constraint
%     forall(i, k in JOB where i < k /\ identical_jobs(i,k)) (
%       exists(j in TASK)(
%         s[i,j] < s[k,j] /\ forall(h in 1..j-1)( s[i,h] = s[k,h] )
%       )
%     );

% ======================= Búsqueda =============================
% Aplanamos s para facilitar la estrategia de búsqueda
array[1..jobs*tasks] of var 0..total: BRANCH_VARS =
  [ s[i,j] | i in JOB, j in TASK ];

%[H1] Heurística 1 — clásica
% solve :: int_search(BRANCH_VARS, first_fail, indomain_min, complete) minimize w;

%[H2] Heurística 2 — informada por conflictos
% solve :: int_search(BRANCH_VARS, dom_w_deg, indomain_min, complete) minimize w;

%[H3] Heurística 3 — orden de entrada
% solve :: int_search(BRANCH_VARS, input_order, indomain_split, complete) minimize w;   

% [D] Default del solver
solve minimize w;

% ======================= Salida ===============================
output
  ["w = \(w)\n"]
  ++
  [ show_int(digs, s[i,j]) ++ " " ++
    if j == tasks then "\n" else "" endif
  | i in JOB, j in TASK ];