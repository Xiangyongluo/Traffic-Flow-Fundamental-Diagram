# -*- coding: utf-8 -*-
# Citation: Cheng, Q., Liu, Z., Lin, Y., Zhou, X., 2021. An s-shaped three-parameter (S3) traffic stream model with consistent car following relationship. Under review.

import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fundamental_diagram_models import Fundamental_Diagram, Estimated_Value, Theoretical_Value, GetMetrics
from scipy.optimize import minimize, Bounds
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'stix'


# Measure running time of the function
def func_running_time(func):
    def inner(*args, **kwargs):
        print(f'INFO Begin to run function: {func.__name__} …')
        time_start = datetime.now()
        res = func(*args, **kwargs)
        time_diff = datetime.now() - time_start
        print(f'INFO Finished running function: {func.__name__}, total: {time_diff.seconds}s')
        print()
        return res
    return inner


def get_outer_layer_data(df: pd.DataFrame, base_col_name: str, target_col_name: str, base_interval: int = 1, percentile: float = 1) -> pd.DataFrame:
    # TDD development
    if {base_col_name, target_col_name}.issubset(df.columns):
        raise Exception(f'ERROR: {base_col_name} or {target_col_name} is not in the dataframe columns!')

    # get min and max values of the base column
    base_col_min, base_col_max = math.floor(df[base_col_name].min()), math.floor(df[base_col_name].max())

    # create criteria for filtering data
    # for each interval, get the mask values
    masks_list = [df[base_col_name].between(i, j) for i, j in zip(range(base_col_min, base_col_max, base_interval),
                                                                range(base_col_min + base_interval, base_col_max, base_interval))]

    # get the target column values for each interval baded on the mask values
    target_values = [df[target_col_name][i].max() * percentile for i in masks_list]
    return pd.DataFrame({base_col_name: range(base_col_min, base_col_max-base_interval), target_col_name: target_values})


class FD_CalibrateSolver:

    def __init__(self, data: pd.DataFrame, model_parameters_dict: dict = {}, **kwargs):
        """This class is used to calibrate the fundamental diagram model.

        Args:
            data (pd.DataFrame): include the flow, density, and speed data. with columns: Flow, Density, Speed.

            model_parameters_dict (dict): a dictionary of model parameters: vf, vc, kc, m, q_max
                example: model_parameters_dict = {"vf": 30, "vc": 30, "kc": 0.5, "m": 4, "q_max": 2000}
        """

        # TDD development
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The data should be a pandas DataFrame.")

        if not {"flow", "density", "speed"}.issubset(data.columns):
            raise ValueError("The data should include the columns: Flow, Density, and Speed.")

        # if not specified the model parameters, then use the default values
        if not model_parameters_dict:
            model_parameters_dict = {"vf": 72.78, "vc": 45.73, "kc": 33.98, "m": 2.98, "q_max": 1554}
            print(f"Info: the default model parameters are used: {model_parameters_dict}\n")

        if not isinstance(model_parameters_dict, dict):
            raise TypeError("The model_parameters_dict should be a dictionary.")

        if not {"vf", "vc", "kc", "m", "q_max"}.issubset(model_parameters_dict.keys()):
            raise KeyError("The model_parameters_dict should at least contain vf, vc, kc, m, q_max.")

        self.speed = np.array(data.speed)
        self.density = np.array(data.density)
        self.flow = np.array(data.flow)
        self.model_parameters_dict = model_parameters_dict

        # initial functions
        self.fd_model_dict = Fundamental_Diagram().fundamental_diagram_func_dict()
        self.estimated_value_dict = Estimated_Value().estimated_value_func_dict()
        self.theoretical_value_dict = Theoretical_Value().theoretical_value_func_dict()
        self.__init_boundary_and_x0()

        # update or add new parameters from kwargs
        if kwargs:
            {setattr(self, key, value) for key, value in kwargs.items()}

    def __create_lower_upper_bounds(self, value: float, oscillation: float = 0.4, user_defined_bound: list = []) -> list:
        return user_defined_bound or [value * (1 - oscillation), value * (1 + oscillation)]

    def __init_boundary_and_x0(self) -> None:

        # free flow speed lower and upper boundary
        lb_vf, ub_vf = self.__create_lower_upper_bounds(self.model_parameters_dict.get("vf"))

        # critical speed lower and upper boundary
        lb_vc, ub_vc = self.__create_lower_upper_bounds(self.model_parameters_dict.get("vc"))

        # give the lower and upper boundary for jam speed
        lb_vjam, ub_vjam = [0, 10]

        # critical density lower and upper boundary
        lb_kc, ub_kc = self.__create_lower_upper_bounds(self.model_parameters_dict.get("kc"))

        # flatness value lower and upper boundary
        lb_m, ub_m = self.__create_lower_upper_bounds(self.model_parameters_dict.get("m"), user_defined_bound=[0, 10])

        # get jam density from critical density and generate lower and upper boundary
        kjam_greenshields = 2 * self.model_parameters_dict["kc"]
        kjam_greenberg = math.e * self.model_parameters_dict.get("kc")
        kjam = kjam_greenshields # for models other than Greenberg and Greenshields

        lb_kjam_greenshields, ub_kjam_greenshields = self.__create_lower_upper_bounds(kjam_greenshields, user_defined_bound=[50, 70])
        lb_kjam_greenberg, ub_kjam_greenberg = self.__create_lower_upper_bounds(kjam_greenberg)
        lb_kjam, ub_kjam = self.__create_lower_upper_bounds(kjam)

        # maximum flow lower and upper boundary
        lb_q_max, ub_q_max = self.__create_lower_upper_bounds(self.model_parameters_dict.get("q_max"))

        print("Info: the lower and upper bounds are:")
        print(f" vf: {lb_vf}, {ub_vf} \n vc: {lb_vc}, {ub_vc}\n vjam: {lb_vjam}, {ub_vjam} \n kc: {lb_kc}, {ub_kc} \n m: {lb_m}, {ub_m} \n kjam: {lb_kjam}, {ub_kjam} \n kjam_greenshields: {lb_kjam_greenshields}, {ub_kjam_greenshields} \n kjam_greenberg:{lb_kjam_greenberg}, {ub_kjam_greenberg} \n q_max: {lb_q_max}, {ub_q_max}\n")

        # crate lower and upper boundary for each model
        boundary_dict = {"S3"          : ([lb_vf, lb_kc, lb_m], [ub_vf, ub_kc, ub_m]),
                         "Greenshields": ([lb_vf, lb_kjam_greenshields], [ub_vf, ub_kjam_greenshields]),
                         "Greenberg"   : ([lb_vc, lb_kjam_greenberg], [ub_vc, ub_kjam_greenberg]),
                         "Underwood"   : ([lb_vf, lb_kc], [ub_vf, ub_kc]),
                         "NF"          : ([lb_vf, lb_kjam, 0], [ub_vf, ub_kjam, 5000]),
                         "GHR_M1"      : ([lb_vf, lb_kc], [ub_vf, ub_kc]),
                         "GHR_M2"      : ([lb_vf, lb_kjam, lb_m], [ub_vf, ub_kjam, ub_m]),
                         "GHR_M3"      : ([lb_vf, lb_kjam, lb_m], [ub_vf, ub_kjam, ub_m]),
                         "KK"          : ([lb_vf, lb_kc, 0, 0, 0], [ub_vf, ub_kc, np.inf, np.inf, np.inf]),
                         "Jayakrishnan": ([lb_vf, lb_vjam, lb_kjam, lb_m], [ub_vf, ub_vjam, 200, ub_m]),
                         "Van_Aerde"   : ([lb_vf, lb_vc, lb_kjam, lb_q_max], [ub_vf, ub_vc, ub_kjam, ub_q_max]),
                         "MacNicholas" : ([lb_vf, lb_kjam, lb_m, 0], [ub_vf, ub_kjam, ub_m, np.inf]),
                         "Wang_3PL"    : ([lb_vf, lb_kc, 1], [ub_vf, ub_kc, 20]),
                         "Wang_4PL"    : ([lb_vf, lb_vjam, lb_kc, 1], [ub_vf, ub_vjam, ub_kc, 20]),
                         "Wang_5PL"    : ([lb_vf, lb_vjam, lb_kc, 1, 0], [ub_vf, ub_vjam, ub_kc, 20, 1]),
                         "Ni"          : ([lb_vf, -1, 0, 0], [ub_vf, 0, 3, 10]),
                        }

        # crate lower and upper boundary for each joint_estimation model
        boundary_joint_estimation_dict = {f"{model_name}_joint_estimation": boundary_dict[model_name] for model_name in boundary_dict}

        # final boundary dictionary
        bounds_dict = boundary_dict | boundary_joint_estimation_dict
        self.bounds = {model_name: Bounds(bounds_dict[model_name][0], bounds_dict[model_name][1]) for model_name in bounds_dict}

        # update bounds for specification test
        self.bounds["Underwood"] = Bounds([lb_vf, 25], [ub_vf, 40])

        # initial x0 for each model
        self.x0 = {model_name:[sum(lb_ub_pair)/2 for lb_ub_pair in zip(bounds_dict.get(model_name)[0], bounds_dict.get(model_name)[1])] for model_name in bounds_dict}

        # update x0 for specification test
        self.x0["KK"] = self.x0["KK"][:2] + [0.25, 0.06, 3.72e-06]
        self.x0["Jayakrishnan"] = self.x0["Jayakrishnan"][:3] + [2]
        self.x0["MacNicholas"] = self.x0["MacNicholas"][:2] + [3, 2]
        self.x0["GHR_M3"] = self.x0["GHR_M3"][:2] + [2]
        self.x0["Ni"] = self.x0["Ni"][:1] + [-3.48e-6, 1/3600, 7.5e-3]
        self.x0["Wang_4PL"] = self.x0["Wang_4PL"][:3] + [1]
        self.x0["Wang_5PL"] = self.x0["Wang_5PL"][:3] + [1, 1]



        print(f"Info: the initial x0 are: {self.x0}\n")

    def _calculate_solution_single(self, model_str: str, init_solution: dict):
        # calibration
        objective = self.fd_model_dict[model_str]
        bound = self.bounds[model_str]
        solution = minimize(objective, init_solution, args=(self.flow, self.density, self.speed), method='trust-constr', bounds=bound)

        return solution.x

    def _calculate_estimated_value_single(self, model_str: str, parameters: dict):
        if model_str in {"Van_Aerde", "Ni", "Van_Aerde_joint_estimation", "Ni_joint_estimation", "S3", "Greenshields", "Greenberg", "Underwood", "NF", "GHR_M1", "GHR_M2", "GHR_M3", "KK", "Jayakrishnan", "MacNicholas", "Wang_3PL", "Wang_4PL", "Wang_5PL", "S3_joint_estimation", "Greenshields_joint_estimation", "Greenberg_joint_estimation", "Underwood_joint_estimation", "NF_joint_estimation", "GHR_M1_joint_estimation", "GHR_M2_joint_estimation", "GHR_M3_joint_estimation", "KK_joint_estimation", "Jayakrishnan_joint_estimation", "MacNicholas_joint_estimation", "Wang_3PL_joint_estimation", "Wang_4PL_joint_estimation", "Wang_5PL_joint_estimation"}:
            # Calculate the estimated flow/speed/density based on the fundamental diagram model.
            # Based on these estimated values, we can further calculate the metrics, such as MRE, MSE, RMSE, etc.
            estimated_values_model_func = self.estimated_value_dict[model_str]
            estimated_density = estimated_values_model_func(parameters, self.flow, self.density, self.speed)[0]
            estimated_flow = estimated_values_model_func(parameters, self.flow, self.density, self.speed)[1]
            return estimated_density, estimated_flow
        else:
            raise ValueError("The model name is not correct, please re-check the model name or define the fundamental diagram model before calibration.")

    @func_running_time
    def get_solutions_dict(self, model_name_list: list = []) -> dict:

        # if not specified, calculate all models
        if not model_name_list:
            return {model_name: self._calculate_solution_single(model_name, self.x0.get(model_name)) for model_name in self.x0}

        # if specified model_name_list, model names not pre-defined, then raise error
        if model_name_list and (not set(model_name_list).issubset(self.x0.keys())):
            raise ValueError("The model name is not correct, please re-check the model name or define the fundamental diagram model before calibration.")

        # if specified the model_name_list, model names defined, then calculate the solutions from the model_name_list
        if model_name_list and set(model_name_list).issubset(self.x0.keys()):
            return {model_name: self._calculate_solution_single(model_name, self.x0.get(model_name)) for model_name in model_name_list}

    @func_running_time
    def get_estimated_value_dict(self, solution_dict: dict, model_name_list: list = []) -> dict:

        # if not specified the model_name_list, then calculate all the estimated values
        if not model_name_list:
            # get estimated flow and speed values for each model
            flow_est = {model_name: self._calculate_estimated_value_single(
                model_name, solution_dict[model_name])[1] for model_name in solution_dict}
            speed_est = {model_name: self._calculate_estimated_value_single(
                model_name, solution_dict[model_name])[0] for model_name in solution_dict}
            return {"flow_est": flow_est, "speed_est": speed_est}

        # if specified the model_name_list, model names not defined, then raise error
        if model_name_list and (not model_name_list.issubset(self.x0.keys())):
            raise ValueError("The model name is not correct, please re-check the model name or define the fundamental diagram model before calibration.")

        # if specified the model_name_list, model names defined, then calculate the estimated values for the specified model names
        if model_name_list and model_name_list.issubset(self.x0.keys()):
            # get estimated flow and speed values for each model
            flow_est = {model_name: self._calculate_estimated_value_single(model_name, solution_dict[model_name])[1] for model_name in model_name_list}
            speed_est = {model_name: self._calculate_estimated_value_single(model_name, solution_dict[model_name])[0] for model_name in model_name_list}
            return {"flow_est": flow_est, "speed_est": speed_est}

    @func_running_time
    def get_matrix_value_dict(self, estimated_dict: dict) -> dict:
        # get estimated speed and flow
        estimated_value_dict = estimated_dict
        model_names_all = list(self.x0.keys())

        # conveert observed speed, flow and density to numpy array
        flow_obs = self.flow
        density_obs = self.density
        speed_obs = self.speed

        # get rmse and r2 results
        rmse_speed_list, rmse_flow_list = [], []
        rmse_r2_overall_list = []
        for model_name in model_names_all:
            flow_est_single = np.array(estimated_value_dict["flow_est"].get(model_name))
            speed_est_single = np.array(estimated_value_dict["speed_est"].get(model_name))
            metrics = GetMetrics(flow_obs, density_obs, speed_obs, flow_est_single, speed_est_single)
            rmse_speed, rmse_flow = metrics.RMSE_Small_Range()

            rmse_speed_list.append(rmse_speed)
            rmse_flow_list.append(rmse_flow)
            try:
                rmse_r2_overall_list.append(metrics.RMSE_Overall())
            except Exception:
                rmse_r2_overall_list.append([np.nan] * 4)

        # create dataframe
        df_rmse_speed = pd.DataFrame(rmse_speed_list, index=model_names_all, columns=["RMSE_speed"]*len(rmse_speed_list[0]))
        df_rmse_flow = pd.DataFrame(rmse_flow_list, index=model_names_all, columns=["RMSE_flow"]*len(rmse_flow_list[0]))
        df_rmse_r2_overall = pd.DataFrame(rmse_r2_overall_list, index=model_names_all, columns=["RMSE_speed_overall", "RMSE_flow_overall", "r2_speed_overall", "r2_flow_overall"])

        # save dataframe to csv
        df_rmse_speed.to_csv("rmse_speed.csv")
        df_rmse_flow.to_csv("rmse_flow.csv")
        df_rmse_r2_overall.to_csv("rmse_r2_overall.csv")

        return {"speed_rmse": df_rmse_speed, "flow_rmse": df_rmse_flow, "rmse_r2_overall": df_rmse_r2_overall}

    @func_running_time
    def plot_fd(self, solution_dict: dict, model_name_list: list = [],  base_model_name: str = "S3", output_path: str = "../examples/Figures_s3"):

        # if not specified the model_name_list, then plot all the fundamental diagram (exclude the base model)
        if not model_name_list:
            model_name_list = [model_name for model_name in self.x0.keys() if model_name != base_model_name]

        # check output path exists or not
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.k = np.linspace(0.000001, 140, 70)
        self.v = np.linspace(0.000001, 90, 70)

        # get base model theoretical flow and speed values
        theoretical_values_base = self.theoretical_value_dict[base_model_name]
        theoretical_speed_base = theoretical_values_base(solution_dict.get(base_model_name), self.k)[0]
        theoretical_flow_base = theoretical_values_base(solution_dict.get(base_model_name), self.k)[1]

        # draw plot accordingly
        for model_name in model_name_list:

            # get single model theoretical flow and speed values
            theoretical_value_single = self.theoretical_value_dict[model_name]
            theoretical_density = theoretical_value_single(solution_dict.get(model_name), self.k)[0]
            theoretical_flow = theoretical_value_single(solution_dict.get(model_name), self.k)[1]

            if model_name in ["Van_Aerde", "Ni", "Van_Aerde_joint_estimation", "Ni_joint_estimation"]:
                x1 = theoretical_density
                y1 = theoretical_flow
                x2 = theoretical_density
                y2 = self.v
                x3 = theoretical_flow
                y3 = self.v
            else:
                x1 = self.k
                y1 = theoretical_flow
                x2 = self.k
                y2 = theoretical_density
                x3 = theoretical_flow
                y3 = theoretical_density

            # create q-k diagram
            fig = plt.figure(figsize=(7,5))
            plt.scatter(self.density.flatten(), self.flow.flatten(), s = 3, marker='o', c='r', edgecolors='r', label = 'Observation')
            plt.plot(x1, y1, 'y-', linewidth=3, label = model_name)
            plt.plot(self.k, theoretical_flow_base, 'b--', linewidth=4, label = "S3")
            plt.plot()
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Density (veh/km)', fontsize=16)
            plt.ylabel('Flow (veh/h)', fontsize=16)
            plt.xlim((0, 140))
            plt.ylim((0, 2250))
            plt.legend(loc='upper right', fontsize=14)
            plt.title('Flow vs. density', fontsize=20)
            fig.savefig(f"{output_path}/flow vs density_{model_name}.png", dpi=400, bbox_inches='tight')

            # create v-k diagram
            fig = plt.figure(figsize=(7,5))
            plt.scatter(self.density.flatten(), self.speed.flatten(), s = 3, marker='o', c='r', edgecolors='r', label = 'Observation')
            plt.plot(x2, y2, 'y-', linewidth=3, label = model_name)
            plt.plot(self.k, theoretical_speed_base, 'b--', linewidth=4, label = "S3")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Density (veh/km)', fontsize=16)
            plt.ylabel('Speed (km/h)', fontsize=16)
            plt.xlim((0, 140))
            plt.ylim((0, 90))
            plt.legend(loc='upper right', fontsize=14)
            plt.title('Speed vs. density', fontsize=20)
            fig.savefig(f"{output_path}/speed vs density_{model_name}.png", dpi=400, bbox_inches='tight')

            # create v-q diagram
            fig = plt.figure(figsize=(7,5))
            plt.scatter(self.flow.flatten(), self.speed.flatten(), s = 3, marker='o', c='r', edgecolors='r', label = 'Observation')
            plt.plot(x3, y3, 'y-', linewidth=3, label = model_name)
            plt.plot(theoretical_flow_base, theoretical_speed_base, 'b--', linewidth=4, label = "S3")
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Flow (veh/h)', fontsize=16)
            plt.ylabel('Speed (km/h)', fontsize=16)
            plt.xlim((0,2250))
            plt.ylim((0, 90))
            plt.legend(loc='upper right', fontsize=14)
            plt.title('Speed vs. flow', fontsize=20)
            fig.savefig(f"{output_path}/speed vs flow_{model_name}.png", dpi=400, bbox_inches='tight')

        print(f"Info: Successfully saved figures in {output_path}")

    @func_running_time
    def plot_fd_combo(self, solution_dict: dict, model_name_list: list = [],
                      output_path: str = "../examples/Figures_combo",
                      transparency: float = 1):

        if not model_name_list:
            model_name_list = list(self.x0.keys())

        # check output path exists or not
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.k = np.linspace(0.000001,140,70)
        self.v = np.linspace(0.000001,90,70)

        # calculate theoretical values for other models
        calibrated_result_list = []
        for model_str in model_name_list:

            theoretical_value_single = self.theoretical_value_dict[model_str]

            theoretical_density = theoretical_value_single(solution_dict.get(model_str), self.k)[0]
            theoretical_flow = theoretical_value_single(solution_dict.get(model_str), self.k)[1]

            if model_str in ["Van_Aerde", "Ni", "Van_Aerde_joint_estimation", "Ni_joint_estimation"]:
                x1 = theoretical_density
                y1 = theoretical_flow
                x2 = theoretical_density
                y2 = self.v
                x3 = theoretical_flow
                y3 = self.v
            else:
                x1 = self.k
                y1 = theoretical_flow
                x2 = self.k
                y2 = theoretical_density
                x3 = theoretical_flow
                y3 = theoretical_density

            calibrated_result_list.append([x1, y1, x2, y2, x3, y3])

        fig = plt.figure(figsize=(7, 5))
        plt.scatter(self.density.flatten(), self.flow.flatten(), s=3,
                    marker='o', c='r', edgecolors='r', label='Observation')
        [plt.plot(calibrated_result_list[i][0],  calibrated_result_list[i][1], linewidth=3, label=model_name_list[i], alpha=transparency) for i in range(len(calibrated_result_list))]

        # plt.plot(self.k, theoretical_flow_S3, 'b--', linewidth=4, label="S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=16)
        plt.ylabel('Flow (veh/h)', fontsize=16)
        plt.xlim((0, 140))
        plt.ylim((0, 2250))
        plt.legend(loc='upper right', fontsize=6)
        plt.title('Flow vs. density', fontsize=20)
        fig.savefig(
            f"{output_path}/flow vs density_combo.png", dpi=400, bbox_inches='tight')

        fig = plt.figure(figsize=(7, 5))
        plt.scatter(self.density.flatten(), self.speed.flatten(), s=3,
                    marker='o', c='r', edgecolors='r', label='Observation')
        [plt.plot(calibrated_result_list[i][2],  calibrated_result_list[i][3], linewidth=3, label=model_name_list[i], alpha=transparency) for i in range(len(calibrated_result_list))]

        # plt.plot(self.k, theoretical_speed_S3, 'b--', linewidth=4, label="S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((0, 140))
        plt.ylim((0, 90))
        plt.legend(loc='upper right', fontsize=6)
        plt.title('Speed vs. density', fontsize=20)
        fig.savefig(
            f"{output_path}/speed vs density_combo.png", dpi=400, bbox_inches='tight')

        fig = plt.figure(figsize=(7, 5))
        plt.scatter(self.flow.flatten(), self.speed.flatten(), s=3,
                    marker='o', c='r', edgecolors='r', label='Observation')
        [plt.plot(calibrated_result_list[i][4],  calibrated_result_list[i][5], linewidth=3, label=model_name_list[i], alpha=transparency) for i in range(len(calibrated_result_list))]

        # plt.plot(theoretical_flow_S3, theoretical_speed_S3,'b--', linewidth=4, label="S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Flow (veh/h)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((0, 2250))
        plt.ylim((0, 90))
        plt.legend(loc='upper right', fontsize=6)
        plt.title('Speed vs. flow', fontsize=20)
        fig.savefig(
            f"{output_path}/speed vs flow_combo.png", dpi=400, bbox_inches='tight')

        print(f"Info: Successfully saved figures in {output_path}")


if __name__ == '__main__':

    # Step 0: Prepare your input data path
    file_path_main = "../data/demo_data_traffic_models_adot/adot_formatted_data_6am_8pm_main.csv"
    file_path_ramp = "../data/demo_data_traffic_models_adot/adot_formatted_data_6am_8pm_ramp.csv"

    # Step 1: load q-k-v data and prepare parameters for calibration
    df_input = pd.read_csv(file_path_main)
    model_parameter_dict = {"vf": 72.78, "vc": 45.73, "kc": 33.98, "m": 2.98, "q_max": 1554}

    # # for the ramp lane
    # df_input = pd.read_csv(file_path_ramp)
    # model_parameter_dict = {"vf": 83.35, "vc": 39.88, "kc": 18.93, "m": 1.88, "q_max": 756}

    # Step 2: define the solver
    solver = FD_CalibrateSolver(df_input, model_parameter_dict)

    # Step 2.1 (optional): specify the initial values for the solver
    # You can change x0 values here or go to function: __init_boundary_and_x0 to change these initial values
    # You can change bounds as well in function: __init_boundary_and_x0

    # Step 3: GET soloution, estimation and rmse results dictionary
    solution_dict = solver.get_solutions_dict(model_name_list=["Underwood", "S3"])
    estimated_dict = solver.get_estimated_value_dict(solution_dict)
    # matrix_value_dict = solver.get_matrix_value_dict(estimated_dict)

    # pd.DataFrame(estimated_dict["flow_est"]).to_csv("flow_est_ramp.csv", index=False)
    # pd.DataFrame(estimated_dict["speed_est"]).to_csv("speed_est_ramp.csv", index=False)

    # Step 4: plot the result
    # compare all models with base model and plot the result and save to file
    model_list_fd = ['Greenshields', 'Greenberg', 'Underwood', 'NF', 'GHR_M1', 'GHR_M2', 'GHR_M3']
    solver.plot_fd(solution_dict, model_name_list=model_list_fd[2:3])

    # plot the result of all models and save to file
    # model_list_combo = ['Greenshields', 'Greenberg', 'S3']
    # solver.plot_fd_combo(solution_dict, model_list_combo)