# -*- coding: utf-8 -*-

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

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
        print(
            f'INFO Finished running function: {func.__name__}, total: {time_diff.seconds}s')
        print()
        return res
    return inner


class FundamentalDiagramModel:

    def __init__(self, observed_flow: np.array, observed_density: np.array, observed_speed: np.array):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed

    def S3(self, beta: list):
        vf, kc, foc = beta
        estimated_speed = vf / \
            np.power(1 + np.power((self.observed_density / kc), foc), 2 / foc)
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj

    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        estimated_speed = vf * (np.tanh((1 - self.observed_density * veh_length) / (
            self.observed_density * transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor))
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj

    def METANET(self, beta):
        vf, kc, foc = beta
        estimated_speed = vf * \
            np.exp(-1/foc*np.power(self.observed_density/kc, foc))
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj


class FirstOrderDerivative:

    def __init__(self, observed_flow: np.array, observed_density: np.array, observed_speed: np.array):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed

    def S3(self, beta: list):
        vf, kc, foc = beta

        intermediate_variable = np.power(self.observed_density / kc, foc)

        first_order_derivative_1 = 2 * np.mean((vf / np.power(1 + intermediate_variable, 2 / foc) - self.observed_speed) / np.power(1 + intermediate_variable, 2 / foc))

        first_order_derivative_2 = 2 * np.mean((vf / np.power(1 + intermediate_variable, 2 / foc) - self.observed_speed) * 2 * vf * intermediate_variable / kc / np.power(1 + intermediate_variable, (foc+2)/foc))

        first_order_derivative_3 = 2 * np.mean((vf / np.power(1 + intermediate_variable, 2 / foc) - self.observed_speed) * 2 * vf * ((1 + intermediate_variable) * np.log(1 + intermediate_variable) - foc * intermediate_variable * np.log(intermediate_variable)) / np.power(foc, 2) / np.power(1 + intermediate_variable, (foc+2) / foc))

        first_order_derivative = np.asarray([first_order_derivative_1, first_order_derivative_2, first_order_derivative_3])

        return first_order_derivative

    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        intermediate_variable = (
            1-self.observed_density*veh_length)/(self.observed_density*transition_width)
        first_order_derivative_1 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(
            form_factor))/(1+np.tanh(form_factor)) - self.observed_speed) * (np.tanh(intermediate_variable) + np.tanh(form_factor))/(1+np.tanh(form_factor)))
        first_order_derivative_2 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(
            form_factor)) - self.observed_speed) * (-1) * vf * np.power(1/np.cosh(intermediate_variable), 2)/(transition_width*np.tanh(form_factor) + transition_width))
        first_order_derivative_3 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor)) - self.observed_speed) * vf * (
            self.observed_density*veh_length-1)*np.power(1/np.cosh(intermediate_variable), 2)/np.power(self.observed_density, 2)/np.power(transition_width, 2)/(1+np.tanh(form_factor)))
        first_order_derivative_4 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(
            form_factor)) - self.observed_speed) * (-1) * vf * np.power(1/np.cosh(form_factor), 2) * (np.tanh(intermediate_variable) - 1)/np.power(1 + np.tanh(form_factor), 2))
        first_order_derivative = np.asarray(
            [first_order_derivative_1, first_order_derivative_2, first_order_derivative_3, first_order_derivative_4])
        return first_order_derivative


class EstimatedValue:

    def __init__(self, observed_flow: np.array, observed_density: np.array, observed_speed: np.array):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed

    def S3(self, beta: list):
        vf, kc, foc = beta
        estimated_speed = vf / np.power(1 + np.power((self.observed_density / kc), foc), 2 / foc)
        estimated_flow = self.observed_density * estimated_speed
        return estimated_speed, estimated_flow

    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        estimated_speed = vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor))
        estimated_flow = self.observed_density*estimated_speed
        return estimated_speed, estimated_flow


class TheoreticalValue:

    def __init__(self, density: np.array):
        self.density = density

    def S3(self, beta: list):
        vf, kc, foc = beta
        theoretical_speed = vf / np.power(1 + np.power((self.density / kc), foc), 2 / foc)
        theoretical_flow = self.density * theoretical_speed
        return theoretical_speed, theoretical_flow

    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        theoretical_speed = vf*(np.tanh((1-self.density*veh_length)/(self.density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor))
        theoretical_flow = self.density*theoretical_speed
        return theoretical_speed, theoretical_flow

    def IDM(self, beta):
        delta = 4
        vf, g0, T0, L = beta
        theoretical_density = 1000*np.sqrt(1-np.power(self.speed/vf, delta))/(g0+self.speed*T0/3.6+L*np.sqrt(1-np.power(self.speed/vf, delta)))
        theoretical_flow = theoretical_density*self.speed
        return theoretical_density, theoretical_flow


class AdamOptimization:

    def __init__(self, objective, first_order_derivative, bounds, x0):
        self.n_iter = 5000
        self.alpha = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.objective = objective
        self.first_order_derivative = first_order_derivative
        self.bounds = bounds
        self.x0 = x0

    def adam(self):
        # keep track of solutions and scores
        solutions = []
        scores = []
        # generate an initial point
        x = list(self.x0)
        score = self.objective(x)
        # initialize first and second moments
        m = [0.0 for _ in range(self.bounds.shape[0])]
        v = [0.0 for _ in range(self.bounds.shape[0])]
        # run the gradient descent updates
        for t in range(1, self.n_iter):
            # calculate gradient g(t)
            g = self.first_order_derivative(x)
            # build a solution one variable at a time
            for i in range(self.bounds.shape[0]):
                # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g[i]
                # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g[i]**2
                # mhat(t) = m(t) / (1 - beta1(t))
                mhat = m[i] / (1.0 - self.beta1**(t+1))
                # vhat(t) = v(t) / (1 - beta2(t))
                vhat = v[i] / (1.0 - self.beta2**(t+1))
                # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
                x[i] = x[i] - self.alpha * mhat / (sqrt(vhat) + self.eps)
            # evaluate candidate point
            score = self.objective(x)
            # keep track of solutions and scores
            solutions.append(x.copy())
            scores.append(score)
            # report progress
        # print('Solution: %s, \nOptimal function value: %.5f' %(solutions[np.argmin(scores)], min(scores)))
        return solutions, scores

    def plot_iteration_process_adam(self, solutions):
        # sample input range uniformly at 0.1 increments
        xaxis = np.arange(self.bounds[0, 0], self.bounds[0, 1], 0.1)
        yaxis = np.arange(self.bounds[1, 0], self.bounds[1, 1], 0.1)
        x, y = np.meshgrid(xaxis, yaxis)
        results = self.objective(x, y)
        solutions = np.asarray(solutions)
        fig, ax = plt.subplots(figsize=(10, 6))
        cs = ax.contourf(x, y, results, levels=50, cmap='jet')
        ax.set_xlim(self.bounds[0, 0], self.bounds[0, 1])
        ax.set_ylim(self.bounds[1, 0], self.bounds[1, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        plt.tick_params(labelsize=14)
        plt.plot(solutions[:, 0], solutions[:, 1], '.-', color='k')
        plt.colorbar(cs)
        plt.title('Iteration process')
        fig.savefig('../Figures/Case 1/Iteration process.png',
                    dpi=300, bbox_inches='tight')


class PlotCalibrationResults:

    def __init__(self, observed_flow: np.array, observed_density: np.array, observed_speed: np.array, calibrated_paras: dict, output_path: str = "../examples/Figures_s3"):

        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed

        self.calibrated_paras_S3 = calibrated_paras["S3"]      # Calibrated from fundamental diagram model, vf, kc, foc
        self.k = np.linspace(0.000001, 150,70)

        self.theoretical_value = TheoreticalValue(self.k)
        self.theoretical_speed_S3, self.theoretical_flow_S3 = self.theoretical_value.S3(self.calibrated_paras_S3)

        self.output_path = output_path

    def plot_qk(self, **kwargs):

        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_density, self.observed_flow, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.k, self.theoretical_flow_S3, 'b-', linewidth=4, label = "S3")

        plt.legend(loc='upper right', fontsize=14)
        plt.title('Flow vs. density', fontsize=20)
        plt.xlabel('Density (veh/mi/ln)', fontsize=16)
        plt.ylabel('Flow (veh/h/ln)', fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(kwargs.get("xlim")) if kwargs.get("xlim") else plt.xlim((0, 150))
        plt.ylim(kwargs.get("ylim")) if kwargs.get("ylim") else plt.ylim((0, 2100))

        fig.savefig(f'{self.output_path}/flow vs density.png', dpi=400, bbox_inches='tight')
        print(f"Info: Successfully saved the figure to {self.output_path}/flow vs density.png")

    def plot_vk(self, **kwargs):

        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_density, self.observed_speed, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.k, self.theoretical_speed_S3, 'b-', linewidth=4, label = "S3")

        plt.title('Speed vs. density', fontsize=20)
        plt.legend(loc='upper right', fontsize=14)
        plt.xlabel('Density (veh/mi/ln)', fontsize=16)
        plt.ylabel('Speed (mi/h)', fontsize=16)

        plt.xlim(kwargs.get("xlim")) if kwargs.get("xlim") else plt.xlim((0, 150))
        plt.ylim(kwargs.get("ylim")) if kwargs.get("ylim") else plt.ylim((0, 90))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        fig.savefig(f'{self.output_path}/speed vs density.png', dpi=400, bbox_inches='tight')
        print(f"Info: Successfully saved the figure to {self.output_path}/speed vs density.png")

    def plot_vq(self, **kwargs):

        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_flow, self.observed_speed, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.theoretical_flow_S3, self.theoretical_speed_S3, 'b-', linewidth=4, label = "S3")

        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. flow', fontsize=20)
        plt.xlabel('Flow (veh/h/ln)', fontsize=16)
        plt.ylabel('Speed (mi/h)', fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(kwargs.get("xlim")) if kwargs.get("xlim") else plt.xlim((0, 2100))
        plt.ylim(kwargs.get("ylim")) if kwargs.get("ylim") else plt.ylim((0, 90))

        fig.savefig(f'{self.output_path}/speed vs flow.png', dpi=400, bbox_inches='tight')
        print(f"Info: Successfully saved the figure to {self.output_path}/speed vs flow.png")


class GetMetrics:

    def __init__(self, observed_flow: np.array, observed_density: np.array, observed_speed: np.array, estimated_flow: np.array, estimated_speed: np.array):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        self.estimated_flow = estimated_flow
        self.estimated_speed = estimated_speed

    def RMSE_Overall(self) -> list:
        rmse_speed = mean_squared_error(
            self.observed_speed, self.estimated_speed, squared=False)
        rmse_flow = mean_squared_error(
            self.observed_flow, self.estimated_flow, squared=False)
        r2_speed = r2_score(self.observed_speed, self.estimated_speed)
        r2_flow = r2_score(self.observed_flow, self.estimated_flow)
        return [rmse_speed, rmse_flow, r2_speed, r2_flow]

    def RMSE_Small_Range(self, interval: int = 10) -> list:
        rmse_speed_small_range = []
        rmse_flow_small_range = []
        density_max_value = min(math.ceil(max(self.observed_density) / 10), 10)
        for i in range(density_max_value):
            temp_index = np.where(
                (self.observed_density >= interval*i) & (self.observed_density < interval*(i+1)))
            observed_speed_i = self.observed_speed[temp_index]
            estimated_speed_i = self.estimated_speed[temp_index]
            observed_flow_i = self.observed_flow[temp_index]
            estimated_flow_i = self.estimated_flow[temp_index]
            try:
                rmse_speed_small_range.append(mean_squared_error(
                    observed_speed_i, estimated_speed_i, squared=False))
            except Exception:
                rmse_speed_small_range.append(0)

            try:
                rmse_flow_small_range.append(mean_squared_error(
                    observed_flow_i, estimated_flow_i, squared=False))
            except Exception:
                rmse_flow_small_range.append(0)
        observed_speed_last = self.observed_speed[np.where(
            (self.observed_density >= density_max_value))]
        estimated_speed_last = self.estimated_speed[np.where(
            (self.observed_density >= density_max_value))]
        observed_flow_last = self.observed_flow[np.where(
            (self.observed_density >= density_max_value))]
        estimated_flow_last = self.estimated_flow[np.where(
            (self.observed_density >= density_max_value))]

        try:
            rmse_speed_small_range.append(mean_squared_error(
                observed_speed_last, estimated_speed_last, squared=False))
        except Exception:
            rmse_speed_small_range.append(0)

        try:
            rmse_flow_small_range.append(mean_squared_error(
                observed_flow_last, estimated_flow_last, squared=False))
        except Exception:
            rmse_flow_small_range.append(0)

        return rmse_speed_small_range, rmse_flow_small_range


class Calibrate:

    def __init__(self, flow: np.array, density: np.array, speed: np.array):
        self.flow = flow
        self.density = density
        self.speed = speed
        self.init_model_dict()

    def init_model_dict(self):
        self.model = FundamentalDiagramModel(self.flow, self.density, self.speed)
        self.first_order_derivative = FirstOrderDerivative(self.flow, self.density, self.speed)
        self.model_dict = {"S3": self.model.S3}
        self.derivative = {"S3": self.first_order_derivative.S3}
        self.bounds = {"S3": np.asarray([[70, 80], [40, 50], [1, 8]])}
        self.x0 = {"S3": np.asarray([75, 45, 2.7])}

    @func_running_time
    def getSolution(self, model_str):
        # calibration
        objective = self.model_dict[model_str]
        derivative = self.derivative[model_str]
        bounds = self.bounds[model_str]
        x0 = self.x0[model_str]
        Adam = AdamOptimization(objective, derivative, bounds, x0)
        solutions, scores = Adam.adam()
        parameters = solutions[np.argmin(scores)]
        return parameters


if __name__ == '__main__':

    # Step 0: Prepare input data path
    path_input = r"../data/demo_data_traffic_models_1/Reading.csv"

    # Step 1: Read data
    df_input = pd.read_csv(path_input)

    # Step 1.1 check if required columns in the dataframe
    if not {"Flow", "Density", "Speed"}.issubset(df_input.columns):
        raise ValueError("Input dataframe must include capitalized columns: Flow, Density, Speed")

    # Step 2: Get Flow, Density, Speed data accordingly
    # Step 2.1 data preprocessing, not necessary, only if you have date and time columns in your input data
    if {"date"}.issubset(df_input.columns):
        date_invalid = list(df_input[df_input["Flow"] == 0]["date"].unique())
        df_input = df_input[~df_input['date'].isin(date_invalid)]

    # Step 2.2 get flow, density and speed data
    flow = np.array(df_input.Flow)
    density = np.array(df_input.Density)
    speed = np.array(df_input.Speed)

    # Step 3: Calibrate
    solver = Calibrate(flow, density, speed)
    result = {"S3": solver.getSolution("S3")}

    # Step 4: Print results
    vf = result['S3'][0]
    kc = result['S3'][1]
    m = result['S3'][2]
    q_max = kc * vf / np.power(2, 2 / m)
    vc = vf / np.power(2, 2 / m)

    print('Calibration results:\n' + 'vf =', format(vf, ".2f") + ' mi/h')
    print('kc =', format(kc, ".2f") + ' veh/mi/ln')
    print('m =', format(result['S3'][2], '.2f'))
    print("Vc = ", format(vc, ".2f") + " mi/h")
    print("q_max = ", format(q_max, ".2f") + " veh/h/ln")

    # Step 5: Plot results
    # plot_results = PlotCalibrationResults(flow, density, speed, result)
    # plot_results.plot_qk(xlim=(0, 60))
    # plot_results.plot_vk(xlim=(0, 60))
    # plot_results.plot_vq(ylim=(10, 80))

    # Step 6: Get metrics
    estimated_value = EstimatedValue(flow, density, speed)
    estimated_speed, estimated_flow = estimated_value.S3(result["S3"])

    metrics = GetMetrics(flow, density, speed, estimated_flow, estimated_speed)
    S3_RMSE_SPEED_Overall, S3_RMSE_FLOW_Overall, S3_R2_SPEED_Overall, S3_R2_FLOW_Overall = metrics.RMSE_Overall()

    print("Job done!")
