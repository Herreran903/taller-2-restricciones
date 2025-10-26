int: jobs;                                    % no of jobs
set of int: JOB = 1..jobs;
int: tasks;                                   % no of tasks per job
set of int: TASK = 1..tasks;
array [JOB,TASK] of int: d;                   % task durations
array [JOB] of int: weights;
array [JOB] of int: due_dates;
int: total = sum(i in JOB, j in TASK)(d[i,j]);% total duration
int: digs = ceil(log(10.0,total));            % digits for output
array [JOB,TASK] of var 0..total: s;          % start times
array [JOB] of var 0.. total: end_job;        % end times
array[JOB] of var 0..total: T;                % tardiness
var 0..total: end;                            % total end time
var int: w;

% nooverlap
predicate no_overlap(var int:s1, int:d1, var int:s2, int:d2) =
    s1 + d1 <= s2 \/ s2 + d2 <= s1;

constraint %% ensure the tasks occur in sequence
    forall(i in JOB) (
        forall(j in 1..tasks-1) 
            (s[i,j] + d[i,j] <= s[i,j+1]) /\
        s[i,tasks] + d[i,tasks] <= end
    );

constraint %% ensure no overlap of tasks
    forall(j in TASK) (
        forall(i,k in JOB where i < k) (
            no_overlap(s[i,j], d[i,j], s[k,j], d[k,j])
        )
    );

constraint forall(j in JOB)(
  end_job[j] = s[j, tasks] + d[j, tasks]
);

constraint forall(j in JOB)(
  T[j] >= end_job[j] - due_dates[j] /\ T[j] >= 0
);

w = sum(j in JOB)( weights[j] * T[j]);
solve minimize w;

%solve :: seq_search([
%             int_search(s, smallest, indomain_min),
%             int_search([end], input_order, indomain_min)])
%      minimize end;


%solve :: seq_search([
%             int_search([end], input_order, indomain_min),
%             int_search(s, smallest, indomain_min)])
%      minimize end;

output ["end = \(w)\n"] ++
       [ show_int(digs,s[i,j]) ++ " " ++ 
         if j == tasks then "\n" else "" endif |
         i in JOB, j in TASK ];
