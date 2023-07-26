//////////// THIS FILE MUST BE CREATED BY EXECUTING WriteAttributeDefinitionFile ////////////
//////////// DO NOT MODIFY THIS FILE MANUALLY ///////////////////////////////////////////////

#ifndef __NOMAD_4_3_DISPLAYATTRIBUTESDEFINITION__
#define __NOMAD_4_3_DISPLAYATTRIBUTESDEFINITION__

_definition = {
{ "DISPLAY_ALL_EVAL",  "bool",  "false",  " Flag to display all evaluations ",  " \n  \n . If true, more points are displayed with parameters DISPLAY_STATS and \n   STATS_FILE \n  \n . If false, only the successful evaluations are displayed. \n  \n . Overrides parameters DISPLAY_INFEASIBLE and DISPLAY_UNSUCCESSFUL \n  \n . Points of the phase one with EB constraint are not displayed \n  \n . Argument: one boolean \n  \n . Example: DISPLAY_ALL_EVAL yes \n  \n . Default: false\n\n",  "  basic display displays stat stats eval evals evaluation evaluations   "  , "false" , "true" , "true" },
{ "DISPLAY_DEGREE",  "int",  "2",  " Level of verbose during execution ",  " \n  \n . Argument: one integer: \n     . 0: No display. Print nothing. \n     . 1: High-level display. Print only errors and results. \n     . 2: Normal display. Print medium level like global information and useful information. \n     . 3: Info-level display. Print lots of information. \n     . 4: Debug-level display. \n     . 5: Even more display. \n  \n . Example: \n     DISPLAY_DEGREE 2    # normal display \n  \n . Default: 2\n\n",  "  basic display verbose output outputs info infos  "  , "false" , "true" , "true" },
{ "DISPLAY_HEADER",  "size_t",  "40",  " Frequency at which the stats header is displayed ",  " \n  \n . Every time this number of stats lines is displayed, the stats header is \n   displayed again. This parameter is for clarity of the display. \n  \n . Value of INF means to never display the header. \n  \n . Default: 40\n\n",  "  advanced  "  , "false" , "true" , "true" },
{ "DISPLAY_INFEASIBLE",  "bool",  "true",  " Flag to display infeasible ",  " \n  \n . When true, do display iterations (standard output and stats file) for which \n   constraints are infeasible. \n  \n . When false, only display iterations where the point is feasible. Except \n the initial point that is always displayed. \n  \n . Adjust this parameter to your needs along with DISPLAY_UNSUCCESSFUL. \n  \n . Argument: one boolean \n  \n . Example: DISPLAY_INFEASIBLE false \n  \n . Default: true\n\n",  "  advanced display displays infeasible  "  , "false" , "true" , "true" },
{ "DISPLAY_MAX_STEP_LEVEL",  "size_t",  "20",  " Depth of the step after which info is not printed ",  " \n . If a step has more than this number of parent steps, it will not be printed. \n  \n . Only has effect when DISPLAY_DEGREE = FULL. \n  \n . Default: 20\n\n",  "  advanced  "  , "false" , "true" , "true" },
{ "DISPLAY_STATS",  "NOMAD::ArrayOfString",  "BBE OBJ",  " Format for displaying the evaluation points ",  " \n  \n . Format of the outputs displayed at each success (single-objective) \n  \n . Format of the final Pareto front (multi-objective) \n  \n . Displays more points with DISPLAY_ALL_EVAL true \n  \n . Arguments: list of strings possibly including the following keywords: \n     BBE        : blackbox evaluations \n     BBO        : blackbox output \n     BLK_EVA    : block evaluation calls \n     BLK_SIZE   : number of points in the block \n     CACHE_HITS : cache hits \n     CACHE_SIZE : cache size \n     CONS_H     : infeasibility (h) value \n     DIRECTION  : direction that generated this point \n     EVAL       : evaluations (includes cache hits) \n     FEAS_BBE   : feasible blackbox evaluations \n     FRAME_CENTER : point that was used as center when generating this point \n     FRAME_SIZE / DELTA_F : frame size delta_k^f \n     GEN_STEP   : name of the step that generated this point \n     H_MAX      : max infeasibility (h) acceptable \n     INF_BBE    : infeasible blackbox evaluations \n     ITER_NUM   : iteration number in which this evaluation was done \n     LAP        : number of lap evaluations since last reset \n     MESH_INDEX : mesh index \n     MESH_SIZE / DELTA_M : mesh size delta_k^m \n     MODEL_EVAL : number of quad or sgtelib model evaluations since last reset \n     OBJ        : objective function value \n     PHASE_ONE_SUCC: success evaluations during phase one phase \n     REL_SUCC   : relative success feasible evaluations (relative to the previous \n                  evaluation, or relative to the mesh center if there was no \n                  previous evaluation in the same pass) \n     SOL        : current feasible iterate \n     SUCCESS_TYPE: success type for this evaluation, compared with the frame center \n     SURROGATE_EVAL: number of static surrogate evaluations \n     THREAD_ALGO: thread number for the algorithm \n     THREAD_NUM : thread number in which this evaluation was done \n     TIME       : real time in seconds \n     TOTAL_MODEL_EVAL: total number of quad or sgtelib model evaluations \n     USER       : user-defined string \n      \n . All outputs may be formatted using C style \n   (%f, %e, %E, %g, %G, %i, %d with the possibility to specify the display width \n   and the precision) \n   example: %5.2Ef displays f in 5 columns and 2 decimals in scientific notation \n . Do not use quotes \n . The '%' character may be explicitly indicated with '\%' \n  \n . Example: \n     DISPLAY_STATS BBE EVAL ( SOL ) OBJ CONS_H \n     DISPLAY_STATS \%5.2obj \n     # for LaTeX tables: \n     DISPLAY_STATS $BBE$ & ( $%12.5SOL, ) & $OBJ$ \\ \n  \n . Default: BBE OBJ\n\n",  "  basic display displays output outputs stat stats success successes  "  , "false" , "true" , "true" },
{ "DISPLAY_FAILED",  "bool",  "false",  " Flag to display failed evaluation ",  " \n  \n . When true, display evaluations that are not ok (failed). \n  \n . Such evaluations will be displayed as INF value for f and h, but blackbox \n   outputs can be displayed as obtained. \n    \n . This option can be used to show failed evaluations for debugging. By default, \n   failed evaluations are not displayed in stats file. \n    \n . Argument: one boolean ('yes' or 'no') \n  \n . Example: DISPLAY_FAILED yes \n  \n  \n . Default: false\n\n",  "  advanced display displays success successes failed failure failures fail fails  "  , "false" , "true" , "true" },
{ "DISPLAY_UNSUCCESSFUL",  "bool",  "false",  " Flag to display unsuccessful ",  " \n  \n . When true, display iterations even when no better solution is found. \n  \n . When false, only display iterations when a better objective value is found. \n  \n . Argument: one boolean ('yes' or 'no') \n  \n . Example: DISPLAY_UNSUCCESSFUL yes \n  \n  \n . Default: false\n\n",  "  advanced display displays success successes failed failure failures fail fails  "  , "false" , "true" , "true" },
{ "STATS_FILE",  "NOMAD::ArrayOfString",  "",  " The name of the stats file ",  " \n  \n . File containing all successes in a formatted way (similar as DISPLAY_STATS in a file) \n  \n . Displays more points when DISPLAY_ALL_EVAL is true \n  \n . Arguments: one string (file name) and one list of strings (for the format of stats) \n  \n . The seed is added to the file name if \n   ADD_SEED_TO_FILE_NAMES=\'yes\' (default) \n  \n . Example: STATS_FILE log.txt BBE SOL %.2fOBJ \n  \n . Default: Empty string.\n\n",  "  basic stat stats file files name display displays output outputs  "  , "false" , "false" , "true" },
{ "EVAL_STATS_FILE",  "string",  "-",  " The name of the file for stats about evaluations and successes ",  " \n  \n . File containing overall stats information about number of evaluations and successes \n  \n . Arguments: one string for the file name \n  \n . The seed is added to the file name if \n   ADD_SEED_TO_FILE_NAMES=\'yes\' (default) \n  \n . Example: EVAL_STATS_FILE detailedStats.txt \n  \n . No default value.\n\n",  "  basic stat stats file files evaluation evaluations  "  , "false" , "false" , "true" },
{ "SOL_FORMAT",  "NOMAD::ArrayOfDouble",  "-",  " Internal parameter for format of the solution ",  " \n  \n . SOL_FORMAT is computed from BB_OUTPUT_TYPE and GRANULARITY \n   parameters. \n  \n . Gives the format precision for display of SOL. May also be used for \n   other ArrayOfDouble of the same DIMENSION (ex. bounds, deltas). \n  \n . CANNOT BE MODIFIED BY USER. Internal parameter. \n  \n . No default value.\n\n",  "  internal  "  , "false" , "true" , "true" },
{ "OBJ_WIDTH",  "size_t",  "0",  " Internal parameter for character width of the objective ",  " \n  \n . Computed to display the objective correctly when NOMAD is run. \n  \n . CANNOT BE MODIFIED BY USER. Internal parameter. \n  \n . Default: 0\n\n",  "  internal  "  , "false" , "false" , "true" },
{ "HISTORY_FILE",  "std::string",  "",  " The name of the history file ",  " \n  \n . The history file contains all evaluations in a simple format (SOL BBO) \n  \n . Arguments: one string (file name) \n  \n . The seed is added to the file name if \n   ADD_SEED_TO_FILE_NAMES=\'yes\' (default) \n  \n . Example: HISTORY_FILE history.txt \n  \n  \n . Default: Empty string.\n\n",  "  basic history file name display displays output outputs  "  , "false" , "false" , "true" },
{ "SOLUTION_FILE",  "std::string",  "",  " The name of the file containing the best feasible solution ",  " \n  \n . The solution file contains the best feasible incumbent point in a simple \n   format (SOL BBO) \n  \n . Arguments: one string (file name) \n  \n . The seed is added to the file name if \n   ADD_SEED_TO_FILE_NAMES=\'yes\' (default) \n  \n . Example: SOLUTION_FILE sol.txt \n  \n  \n . Default: Empty string.\n\n",  "  basic solution best incumbent file name display displays output outputs  "  , "false" , "false" , "true" } };

#endif
