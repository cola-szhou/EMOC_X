from joblib import Parallel, delayed
import pandas as pd
from typing import List, Optional, Dict, Union, Any, Callable, Tuple
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt

# from emoc.problem import Problem
# from emoc.algorithm import Algorithm
from emoc.core import Global
from emoc.utils import LoadPFData
from emoc.metrics import *
import EMOC


# class EMOC_Manager:
#     """
#     EMOC manager that is more capable for conducting large-scale experiments.
#     Read more in the `User Guide <url>`.

#     Attributes
#     ----------
#     - `.algorithms` : The algorithms for the most recent run.
#     - `.problems` : The problems for the most recent run.
#     - `.metrics` : The current metrics used for evaluation.
#     - `.n_gen` : The number of generations used for optimization.
#     - `.results` : The dictionary containing raw optimization results.

#     """
#     def __init__(
#             self,
#             n_gen: int = 200,
#             metrics: Union[str, list, tuple, dict, Callable] = None,
#     ) -> None:
#         """
#         Basic setups for experiments.

#         Parameters
#         ----------
#         n_gen : int, default=200
#             Number of generations to run for each experiment.

#         metrics : str, list, tuple, dict, or callable, default=None
#             Metrics to evaluate the performance of the optimizer.

#             If `metrics` represents a single metric, one can use:

#             - a single string (see...);
#             - a callable (see...) that returns a single value.

#             If `metrics` represents multiple metrics, one can use:

#             - a list or tuple of unique strings along with names;
#             - a callable returning a dictionary where the keys are the metric
#             names and the values are the metric scores;
#             - a dictionary with metric names as keys and callables a values.
#         """
#         self.n_gen = n_gen
#         self.metrics = metrics
#         self.results = {}

#     def multi_optimize(
#             self,
#             algorithms: Any,
#             problems: Any,
#             n_runs: int = 5,
#             record_X: bool = False,
#             log_interval: int = None,
#             n_jobs: int = -1,
#             verbose: int = 0,
#             pre_dispatch = '2*n_jobs',
#     ) -> pd.DataFrame:
#         """
#         Run the experiment.

#         Parameters
#         ----------
#         algorithms : list of tuple or emoc.Algorithm
#             List of (name, algorithm) tuples (implementing emoc.Algorithm). Single
#             algorithm is also accepted, but will be assigned a default name.

#         problems : list of tuple or emoc.Problem
#             The problem(s) to be tested.

#         n_runs : int, default=5
#             Number of repeated runs for each experiment.

#         record_X : bool, default=False
#             Whether to record the decision vectors during optimization. If True, the
#             corresponding data will be stored in the `.results` dictionary. Note that
#             enabling this feature can be significantly increase memory consumption.

#         log_interval : bool, default=None
#             The interval in generations to record data. If None, only the PF (and X if
#             `record_X` is enabled) at the last generation will be recorded. If, say,
#             `log_interval`=10, then data will be recorded in every 10 generations.

#         n_jobs : int, default=-1
#             Number of jobs to run in parallel. Experiments are are parallelized
#             across all algorithms, problems, and runs (each consitute an experiment
#             instance). ``None`` means 1, while ``-1`` means using all processors.

#         verbose : int, default=0
#             The verbosity level.

#         Returns
#         -------
#         scores : dict of float arrays of shape (n_algorithms, n_problems, n_runs)
#             Array of scores of the optimizer for each experiment.
#         """
#         problems = [("singular problem", problems)] if not isinstance(problems, list) else problems
#         algorithms = [("singular algorithm", algorithms)] if not isinstance(algorithms, list) else algorithms
#         self.metrics = [("metric", self.metrics)] if not isinstance(self.metrics, list) else self.metrics
#         self.problems = [("problem", problems)] if not isinstance(problems, list) else problems
#         self.algorithms = [("algorithm", algorithms)] if not isinstance(algorithms, list) else algorithms

#         self.metric_names = list(value[0] for value in self.metrics)
#         self.problem_names = list(value[0] for value in self.problems)
#         self.algorithm_names = list(value[0] for value in self.algorithms)
#         self.n_runs = n_runs
#         self.log_interval = log_interval if log_interval != None else self.n_gen
#         self.record_X = record_X

#         tasks = [(alg_name, algorithm, prob_name, problem, self.metrics)
#                 for alg_name, algorithm in algorithms
#                 for prob_name, problem in problems
#                 for _ in range(max(1, n_runs))]

#         parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
#         self.results = parallel(delayed(self._optimize_and_evaluate)(*task) for task in tasks)
#         self.results = self._transform_results(self.results)

#     def optimize(
#             self,
#             algorithm,
#             problem,
#             record_X: bool = False,
#             log_interval: int = None,
#         ):
#         """
#         Function for performing single run.
#         """
#         self.log_interval = log_interval if log_interval != None else self.n_gen
#         self.record_X = record_X
#         results = self._optimize_and_evaluate("alg", algorithm, "prob", problem, self.metrics)

#     def report(self, gen = None, orientation = 1) -> pd.DataFrame:
#         """
#         Generate statistical report of the optimization
#         """
#         return self._results_to_dataframe(gen = gen, orientation = orientation)

#     def get_results(self):
#         pass

#     def _transform_results(self, results_dict):

#         transformed_results = {}
#         for result in results_dict:

#             (algorithm, problem), logger = result
#             key = (algorithm, problem)

#             if key not in transformed_results:
#                 transformed_results[key] = {}

#             run_number = len(transformed_results[key])
#             transformed_results[key][run_number] = logger

#         return transformed_results

#     def _optimize_and_evaluate(self, alg_name, algorithm, prob_name, problem, metrics):

#         result = minimize(problem, algorithm, ('n_gen', self.n_gen), verbose=False)
#         step = self.log_interval
#         results = {}
#         for k in range(step, self.n_gen + step, step):
#             results[k] = {}
#             # Record PF data
#             pf = [list(result.history[k - 1].pop[i].F) for i in range(algorithm.pop_size)]
#             results[k]["PF"] = pf
#             # Record decision variables
#             if self.record_X:
#                 results[k]["X"] = [list(result.history[k - 1].pop[i].X) for i in range(algorithm.pop_size)]
#             # Calculate and record metrics
#             metric_results = {}
#             for name, metric in metrics:
#                 instantiated_metric = metric(problem.pareto_front())
#                 metric_result = instantiated_metric(np.array(pf))
#                 metric_results[name] = metric_result
#             results[k]['metrics'] = metric_results

#         return ((alg_name, prob_name), results)

#     def _results_to_dataframe(self, gen = None, orientation = 1):
#         gen = self.n_gen if gen == None else gen
#         if orientation == 1:
#             return self._orientation1(gen)
#         elif orientation == 2:
#             return self._orientation2(gen)
#         elif orientation == 3:
#             return self._orientation3(gen)
#         else:
#             raise ValueError("Invalid orientation value. Use 1, 2 or 3.")

#     def _orientation1(self, gen):

#         row_index = pd.MultiIndex.from_product(
#             [self.problem_names],
#             # names=['Problem']
#         )

#         column_index = pd.MultiIndex.from_product(
#             [self.metric_names, self.algorithm_names, ['avg', 'var']],
#             # names=["Metric", 'Algorithm', 'Stat']
#         )

#         df = pd.DataFrame(index=row_index, columns=column_index).sort_index()

#         metric_values = {
#             problem: {(metric, algorithm): []
#             for algorithm in self.algorithm_names
#             for metric in self.metric_names}
#             for problem in self.problem_names
#             }

#         for (algorithm, problem), runs in self.results.items():
#             for run in runs.values():
#                 for metric, value in run[gen]['metrics'].items():
#                     metric_values[problem][(metric, algorithm)].append(value)

#         for problem, results in metric_values.items():
#             for (metric, algorithm), values in results.items():
#                 avg = sum(values) / len(values) if values else None
#                 var = pd.Series(values).var() if len(values) > 1 else 0
#                 df.loc[problem, (metric, algorithm, 'avg')] = avg
#                 df.loc[problem, (metric, algorithm, 'var')] = var

#         return df

#     def _orientation3(self, gen):

#         row_index = pd.MultiIndex.from_product(
#             [self.problem_names],
#             # names=['Problem']
#         )

#         column_index = pd.MultiIndex.from_product(
#             [self.algorithm_names, self.metric_names, ['avg', 'var']],
#             # names=['Algorithm', "Metric", 'Stat']
#         )

#         df = pd.DataFrame(index=row_index, columns=column_index).sort_index()

#         metric_values = {
#             problem: {(metric, algorithm): []
#             for algorithm in self.algorithm_names
#             for metric in self.metric_names}
#             for problem in self.problem_names
#             }

#         for (algorithm, problem), runs in self.results.items():
#             for run in runs.values():
#                 for metric, value in run[gen]['metrics'].items():
#                     metric_values[problem][(metric, algorithm)].append(value)

#         for problem, results in metric_values.items():
#             for (metric, algorithm), values in results.items():
#                 avg = sum(values) / len(values) if values else None
#                 var = pd.Series(values).var() if len(values) > 1 else 0
#                 df.loc[problem, (algorithm, metric, 'avg')] = avg
#                 df.loc[problem, (algorithm, metric, 'var')] = var

#         return df

#     def _orientation2(self, gen):

#         row_index = pd.MultiIndex.from_product(
#             [self.problem_names, self.metric_names],
#             # names=['Problem', 'Metric']
#         )

#         column_index = pd.MultiIndex.from_product(
#             [self.algorithm_names, ['avg', 'var']],
#             # names=['Algorithm', 'Stat']
#         )

#         df = pd.DataFrame(index=row_index, columns=column_index).sort_index()

#         metric_values = {
#             (problem, metric): {algorithm: []
#                                 for algorithm in self.algorithm_names}
#                                 for problem in self.problem_names
#                                 for metric in self.metric_names
#             }

#         for (algorithm, problem), runs in self.results.items():
#             for run in runs.values():
#                 for metric, value in run[gen]['metrics'].items():
#                     metric_values[(problem, metric)][algorithm].append(value)

#         for (problem, metric), algorithms_values in metric_values.items():
#             for algorithm, values in algorithms_values.items():
#                 avg = sum(values) / len(values) if values else None
#                 var = pd.Series(values).var() if len(values) > 1 else 0
#                 df.loc[(problem, metric), (algorithm, 'avg')] = avg
#                 df.loc[(problem, metric), (algorithm, 'var')] = var

#         return df

#     def plot_metric(
#             self,
#             metric,
#             algorithms = None,
#             problems = None,
#             figsize = (12, 2.5),
#             ncols = 5,
#             nrows = None,
#         ):

#         algorithms = self.algorithm_names if algorithms == None else algorithms
#         problems = self.problem_names if problems == None else problems
#         nrows = len(problems) // 5 + 1 if nrows == None else nrows

#         fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)
#         axs = axs.flatten()

#         colors = ['blue', 'red', 'orange']

#         for i, problem in enumerate(problems):
#             ax = axs[i]

#             for algorithm, color in zip(algorithms, colors):
#                 runs = self.results[(algorithm, problem)].keys()
#                 gens = self.results[(algorithm, problem)][next(iter(runs))].keys()
#                 metric_list = []
#                 for gen in gens:
#                     values = [self.results[(algorithm, problem)][run][gen]["metrics"][metric] for run in runs]
#                     avg = sum(values) / len(values) if values else None
#                     var = pd.Series(values).var() if len(values) > 1 else 0
#                     metric_list.append((avg, var))

#                 data = pd.DataFrame(metric_list, columns=["avg", "var"], index=gens)
#                 upper_bound = data['avg'] + data['var']
#                 lower_bound = data['avg'] - data['var']
#                 lower_bound[lower_bound < 0] = 0

#                 ax.plot(
#                     data.index,
#                     data['avg'],
#                     label = algorithm,
#                     marker = 'o',
#                     markerfacecolor = 'none',
#                     markeredgecolor = color,
#                     color = color
#                 )

#                 ax.fill_between(data.index, lower_bound, upper_bound, alpha=0.2, color = color)

#             ax.set_yscale('log')
#             ax.set_title(f"Problem: {problem}")
#             ax.set_xlabel("Generation")
#             ax.grid(True, linestyle='--', linewidth=0.5, color = 'lightgrey', alpha = 1)
#             ax.tick_params(axis='both', direction='in', which='both')

#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(handles, labels, loc='upper right', bbox_to_anchor = (1.12, 0.8))
#         plt.show()


# class EMOC_Manager_Exp:
#     """
#     EMOC manager that is more capable for conducting large-scale experiments.
#     Read more in the `User Guide <url>`.
#     """
#     def __init__(
#             self,
#             s_pop: int = 100,
#             max_eva: int = 250000,
#             n_runs: int = 1,
#             metrics: Union[str, list, tuple, dict, Callable] = None,
#     ) -> None:
#         """
#         Basic setups for experiments.

#         Parameters
#         ----------
#         s_pop : int, default=100
#             Number of individuals in the population.

#         max_eva : int, default=250000
#             Maximum allowable function evaluations. Defaults to 250000.

#         n_runs : int, default=1
#             Number of repeated runs for each experiment.

#         metrics : str, list, tuple, dict, or callable, default=None
#             Metrics to evaluate the performance of the optimizer.

#             If `metrics` represents a single metric, one can use:

#             - a single string (see...);
#             - a callable (see...) that returns a single value.

#             If `metrics` represents multiple metrics, one can use:

#             - a list or tuple of unique strings along with names;
#             - a callable returning a dictionary where the keys are the metric
#             names and the values are the metric scores;
#             - a dictionary with metric names as keys and callables a values.
#         """
#         self.s_pop_ = s_pop
#         self.max_eva_ = max_eva
#         self.n_runs_ = n_runs
#         self.metrics = metrics
#         self.global_ = []

#     def run(
#             self,
#             algorithm: Any,
#             problem: Any,
#             n_jobs: int = -1,
#             verbose: int = 0,
#             solve_params = None
#     ) -> None:
#         """
#         Run the experiment.

#         Parameters
#         ----------
#         algorithm : list of tuple or emoc.Algorithm
#             List of (name, algorithm) tuples (implementing emoc.Algorithm). Single
#             algorithm is also accepted, but will be assigned a default name.

#         problem : list of tuple or emoc.Problem
#             The problem(s) to be tested.

#         n_jobs : int, default=-1
#             Number of jobs to run in parallel. Experiments are are parallelized
#             across all algorithms, problems, and runs (each consitute an experiment
#             instance). ``None`` means 1, while ``-1`` means using all processors.
# de
#         verbose : int, default=0
#             The verbosity level.

#         solve_params : dict, default=None
#             Parameters to pass to the solve method of the optimizer.

#         Returns
#         -------
#         scores : dict of float arrays of shape (n_algorithms, n_problems, n_runs)
#             Array of scores of the optimizer for each experiment.
#         """

# def _get_all_exp_instances(problem, algorithm, n_runs):
#     pass

# class EMOC_Manager:

#     """
#     Our most straightforward experiment manager for simple tasks.
#     """
#     def __init__(self, population_num=100, max_evaluation=250000, run=1):
#         self.population_num_ = population_num
#         self.max_evaluation_ = max_evaluation
#         self.run_ = run
#         self.global_ = []

#     def run(self, algorithm, problem, multi_threading=False, n_workers=5):
#         for r in range(self.run_):
#             global_ = Global()
#             global_.SetParam(problem.dec_num_, problem.obj_num_, problem.lower_bound_, problem.upper_bound_, self.population_num_, self.max_evaluation_)
#             self.global_.append(global_)

#             if multi_threading:
#                 pool = multiprocessing.Pool(n_workers)
#                 for i in range(n_workers):
#                     pool.apply_async(self.run_algorithm, args=(global_, algorithm, problem, r))
#                 pool.close()
#                 pool.join()
#             else:
#                 self.run_algorithm(global_, algorithm, problem, r)
#     """
#     def experiment_run(self, algorithm:list, problem:list):
#         self.global_.SetParam(problem.dec_num_, problem.obj_num_, problem.lower_bound_, problem.upper_bound_, self.population_num_, self.max_evaluation_, self.output_interval_)
#         for _ in range(self.run_):
#             if isinstance(algorithm, Algorithm):
#                 if isinstance(problem, Problem):
#                     algorithm.solve(problem, self.population_num_, self.max_evaluation_, self.output_interval_)
#                 elif isinstance(problem, list):
#                     for p in problem:
#                         algorithm.solve(p, self.population_num_, self.max_evaluation_, self.output_interval_)
#                 else:
#                     print("The type of problem is wrong!")
#             elif isinstance(algorithm, list):
#                 for al in algorithm:
#                     if isinstance(problem, Problem):
#                         al.solve(problem, self.population_num_, self.max_evaluation_, self.output_interval_)
#                     elif isinstance(problem, list):
#                         for p in problem:
#                             algorithm.solve(p, self.population_num_, self.max_evaluation_, self.output_interval_)
#             else:
#                 print("The type of algorithm is wrong!")
#     """
#     def run_algorithm(self, global_, algorithm, problem, run_id):
#         algorithm.runtime_ = 0.0
#         algorithm.Solve(problem, global_)  # add the problem to global
#         algorithm.PrintResult(run_id)

#     def show_result():
#         pass

#     def plot():
#         pass


class EMOC_Manager:
    # TO-DO: how to deal with metrics? and our results have recorded in self.global_, not results_
    def __init__(self, population_num=100, max_evaluation=25000, run=1):
        EMOC.randomize()
        self.population_num_ = population_num
        self.max_evaluation_ = max_evaluation
        self.run_ = run
        self.global_ = []
        self.metric_history = {}

    def multi_optimize(
        self,
        algorithms: Any,
        problems: Any,
        metric: Any,
        record_X: bool = False,
        n_jobs: int = -1,
        verbose: int = 0,
        pre_dispatch="2*n_jobs",
    ) -> pd.DataFrame:
        self.problems_ = [
            (problems.name, problems) if not isinstance(problems, list) else problems
        ]
        self.algorithms_ = [
            (
                (algorithms.name, algorithms)
                if not isinstance(algorithms, list)
                else algorithms
            )
        ]
        self.metrics_ = (
            [("metric", self.metrics)]
            if not isinstance(self.metrics, list)
            else self.metrics
        )
        self.problem_names_ = list(value[0] for value in self.problems_)
        self.algorithm_names_ = list(value[0] for value in self.algorithms_)

        tasks = [
            (alg_name, alg, prob_name, prob, self.metrics_)
            for alg_name, alg in self.algorithms_
            for prob_name, prob in self.problems_
            for _ in range(max(1, self.run_))
        ]

        # TO-DO: modify it through process bar
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
        self.results_ = parallel(
            delayed(self._optimize_and_evaluate)(*task) for task in tasks
        )
        self.results_ = self._transform_results(self.results_)

    def optimize(
        self,
        algorithm,
        problem,
        metrics,
        output_interval: int = None,
        pf_path: Optional[str] = None,
    ):
        """
        Function for performing single run.
        Args:
            algorithm (Algorithm): _description_
            problem (Problem): _description_
            record_decision_variables (bool, optional): _description_. Defaults to False.
            log_interval (int, optional): _description_. Defaults to None.
        """
        # self.metrics_ = (
        #     [("metric", self.metrics)]
        #     if not isinstance(self.metrics, list)
        #     else self.metrics
        # )
        self.output_interval = (
            output_interval if output_interval != None else self.max_evaluation_
        )
        results = self._optimize_and_evaluate(
            algorithm.name, algorithm, problem.name, problem, pf_path=pf_path
        )

    def _optimize_and_evaluate(
        self, alg_name: str, alg, prob_name: str, prob, pf_path: Optional[str] = None
    ):
        global_ = Global()
        global_.SetParam(
            prob.dec_num_,
            prob.obj_num_,
            prob.dec_space_,
            self.population_num_,
            self.output_interval,
            self.max_evaluation_,
        )
        self.global_.append(global_)
        alg.runtime_ = 0.0
        alg.Solve(prob, self.global_[-1])

        # Load the PF data
        pf = LoadPFData(pf_path, prob.name, prob.obj_num_)
        if pf == []:
            return
        metric_history = {}
        igd_history = []
        igd_plus_history = []
        gd_history = []
        gd_plus_history = []
        hv_history = []
        spacing_history = []

        for i in range(len(self.global_[-1].record_)):
            igd = CalculateIGD(self.global_[-1].record_[i].pop_, pf)
            igd_plus = CalculateIGDPlus(self.global_[-1].record_[i].pop_, pf)
            gd = CalculateGD(self.global_[-1].record_[i].pop_, pf)
            gd_plus = CalculateGDPlus(self.global_[-1].record_[i].pop_, pf)
            spacing = CalculateSpacing(self.global_[-1].record_[i].pop_)
            # hv = CalculateHV(pf, self.global_[-1].record_[i])
            igd_history.append(igd)
            igd_plus_history.append(igd_plus)
            gd_history.append(gd)
            gd_plus_history.append(gd_plus)
            spacing_history.append(spacing)
        metric_history["igd"] = igd_history
        metric_history["igd_plus"] = igd_plus_history
        metric_history["gd"] = gd_history
        metric_history["gd_plus"] = gd_plus_history
        metric_history["spacing"] = spacing_history
        metric_history["hv"] = hv_history
        return (alg_name, prob_name), metric_history
