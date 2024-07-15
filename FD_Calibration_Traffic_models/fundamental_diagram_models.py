# -*- coding: utf-8 -*-

import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score


class Fundamental_Diagram:

    def S3(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        return np.sum(np.power(estimated_speed - observed_speed, 2))

    def Greenshields(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        return np.sum(np.power(estimated_speed - observed_speed, 2))

    def Greenberg(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        return np.sum(np.power(estimated_speed - observed_speed, 2))

    def Underwood(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def NF(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def GHR_M1(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def GHR_M2(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def GHR_M3(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def KK(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Jayakrishnan(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Van_Aerde(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        f_obj = np.sum(np.power(estimated_density - observed_density, 2))
        return f_obj

    def MacNicholas(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Wang_3PL(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Wang_4PL(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Wang_5PL(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2))
        return f_obj

    def Ni(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        f_obj = np.sum(np.power(estimated_density - observed_density, 2))
        return f_obj

    def S3_joint_estimation(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Greenshields_joint_estimation(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Greenberg_joint_estimation(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Underwood_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def NF_joint_estimation(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def GHR_M1_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def GHR_M2_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def GHR_M3_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def KK_joint_estimation(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Jayakrishnan_joint_estimation(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Van_Aerde_joint_estimation(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        estimated_flow = observed_speed * estimated_density
        sigma = np.var(observed_density) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_density - observed_density, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def MacNicholas_joint_estimation(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Wang_3PL_joint_estimation(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Wang_4PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Wang_5PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        estimated_flow = estimated_speed * observed_density
        sigma = np.var(observed_speed) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_speed - observed_speed, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def Ni_joint_estimation(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        estimated_flow = observed_speed * estimated_density
        sigma = np.var(observed_density) / np.var(observed_flow)
        f_obj = np.sum(np.power(estimated_density - observed_density, 2) + sigma * np.power(estimated_flow - observed_flow, 2))
        return f_obj

    def fundamental_diagram_func_dict(self):
        return {
            "S3":self.S3,
            "Greenshields":self.Greenshields,
            "Greenberg":self.Greenberg,
            "Underwood":self.Underwood,
            "NF":self.NF,
            "GHR_M1":self.GHR_M1,
            "GHR_M2":self.GHR_M2,
            "GHR_M3":self.GHR_M3,
            "KK":self.KK,
            "Jayakrishnan":self.Jayakrishnan,
            "Van_Aerde":self.Van_Aerde,
            "MacNicholas":self.MacNicholas,
            "Wang_3PL":self.Wang_3PL,
            "Wang_4PL":self.Wang_4PL,
            "Wang_5PL":self.Wang_5PL,
            "Ni":self.Ni,
            "S3_joint_estimation":self.S3_joint_estimation,
            "Greenshields_joint_estimation":self.Greenshields_joint_estimation,
            "Greenberg_joint_estimation":self.Greenberg_joint_estimation,
            "Underwood_joint_estimation":self.Underwood_joint_estimation,
            "NF_joint_estimation":self.NF_joint_estimation,
            "GHR_M1_joint_estimation":self.GHR_M1_joint_estimation,
            "GHR_M2_joint_estimation":self.GHR_M2_joint_estimation,
            "GHR_M3_joint_estimation":self.GHR_M3_joint_estimation,
            "KK_joint_estimation":self.KK_joint_estimation,
            "Jayakrishnan_joint_estimation":self.Jayakrishnan_joint_estimation,
            "Van_Aerde_joint_estimation":self.Van_Aerde_joint_estimation,
            "MacNicholas_joint_estimation":self.MacNicholas_joint_estimation,
            "Wang_3PL_joint_estimation":self.Wang_3PL_joint_estimation,
            "Wang_4PL_joint_estimation":self.Wang_4PL_joint_estimation,
            "Wang_5PL_joint_estimation":self.Wang_5PL_joint_estimation,
            "Ni_joint_estimation":self.Ni_joint_estimation,
        }


class Estimated_Value:

    def S3(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        estimated_flow = observed_density*vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        return estimated_speed, estimated_flow

    def Greenshields(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        estimated_flow = observed_density*vf*(1 - observed_density/k_jam)
        return estimated_speed, estimated_flow

    def Greenberg(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        estimated_flow = observed_density*vc*np.log(k_jam/observed_density)
        return estimated_speed, estimated_flow

    def Underwood(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        estimated_flow = observed_density*vf*np.exp(-1*observed_density/kc)
        return estimated_speed, estimated_flow

    def NF(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        estimated_flow = observed_density*vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        return estimated_speed, estimated_flow

    def GHR_M1(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        estimated_flow = observed_density*vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        return estimated_speed, estimated_flow

    def GHR_M2(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        estimated_flow = observed_density*vf*(1 - np.power(observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def GHR_M3(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*vf*np.power(1 - observed_density/k_jam, m)
        return estimated_speed, estimated_flow

    def KK(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        estimated_flow = observed_density*vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        return estimated_speed, estimated_flow

    def Jayakrishnan(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*(v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def Van_Aerde(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        estimated_flow = observed_speed/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        return estimated_density, estimated_flow

    def MacNicholas(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        estimated_flow = observed_density*vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        return estimated_speed, estimated_flow

    def Wang_3PL(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*vf/(1+np.exp((observed_density-kc)/theta))
        return estimated_speed, estimated_flow

    def Wang_4PL(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*(vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta)))
        return estimated_speed, estimated_flow

    def Wang_5PL(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        estimated_flow = observed_density*(vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2))
        return estimated_speed, estimated_flow

    def Ni(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        estimated_flow = observed_speed/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        return estimated_density, estimated_flow

    def S3_joint_estimation(self, beta, *args):
        vf, kc, foc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        estimated_flow = observed_density*vf/np.power(1 + np.power((observed_density/kc), foc), 2/foc)
        return estimated_speed, estimated_flow

    def Greenshields_joint_estimation(self, beta, *args):
        vf, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - observed_density/k_jam)
        estimated_flow = observed_density*vf*(1 - observed_density/k_jam)
        return estimated_speed, estimated_flow

    def Greenberg_joint_estimation(self, beta, *args):
        vc, k_jam = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vc*np.log(k_jam/observed_density)
        estimated_flow = observed_density*vc*np.log(k_jam/observed_density)
        return estimated_speed, estimated_flow

    def Underwood_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-1*observed_density/kc)
        estimated_flow = observed_density*vf*np.exp(-1*observed_density/kc)
        return estimated_speed, estimated_flow

    def NF_joint_estimation(self, beta, *args):
        vf, k_jam, lambda_NF = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        estimated_flow = observed_density*vf*(1-np.exp(-1*lambda_NF/vf*(1/observed_density - 1/k_jam)))
        return estimated_speed, estimated_flow

    def GHR_M1_joint_estimation(self, beta, *args):
        vf, kc = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        estimated_flow = observed_density*vf*np.exp(-0.5*np.power(observed_density/kc, 2))
        return estimated_speed, estimated_flow

    def GHR_M2_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1 - np.power(observed_density/k_jam, m))
        estimated_flow = observed_density*vf*(1 - np.power(observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def GHR_M3_joint_estimation(self, beta, *args):
        vf, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*vf*np.power(1 - observed_density/k_jam, m)
        return estimated_speed, estimated_flow

    def KK_joint_estimation(self, beta, *args):
        vf, kc, c1, c2, c3 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        estimated_flow = observed_density*vf*(1/(1 + np.exp((observed_density/kc - c1)/c2)) - c3)
        return estimated_speed, estimated_flow

    def Jayakrishnan_joint_estimation(self, beta, *args):
        vf, v_jam, k_jam, m = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m)
        estimated_flow = observed_density*(v_jam + (vf - v_jam)*np.power(1 - observed_density/k_jam, m))
        return estimated_speed, estimated_flow

    def Van_Aerde_joint_estimation(self, beta, *args):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        estimated_flow = observed_speed/(c1 + c2/(vf-observed_speed) + c3*observed_speed)
        return estimated_density, estimated_flow

    def MacNicholas_joint_estimation(self, beta, *args):
        vf, k_jam, m, c = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        estimated_flow = observed_density*vf*(np.power(k_jam, m) - np.power(observed_density, m))/(np.power(k_jam, m) + c*np.power(observed_density, m))
        return estimated_speed, estimated_flow

    def Wang_3PL_joint_estimation(self, beta, *args):
        vf, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vf/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*vf/(1+np.exp((observed_density-kc)/theta))
        return estimated_speed, estimated_flow

    def Wang_4PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta))
        estimated_flow = observed_density*(vb + (vf-vb)/(1+np.exp((observed_density-kc)/theta)))
        return estimated_speed, estimated_flow

    def Wang_5PL_joint_estimation(self, beta, *args):
        vf, vb, kc, theta1, theta2 = beta
        observed_flow, observed_density, observed_speed = args
        estimated_speed = vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2)
        estimated_flow = observed_density*(vb + (vf-vb)/np.power(1+np.exp((observed_density-kc)/theta1), theta2))
        return estimated_speed, estimated_flow

    def Ni_joint_estimation(self, beta, *args):
        vf, gamma, tao, l = beta
        observed_flow, observed_density, observed_speed = args
        estimated_density = 1/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        estimated_flow = observed_speed/((gamma*np.power(observed_speed,2)+tao*observed_speed+l)*(1-np.log(1-observed_speed/vf)))
        return estimated_density, estimated_flow

    def estimated_value_func_dict(self):
        return {
            "S3":self.S3,
            "Greenshields":self.Greenshields,
            "Greenberg":self.Greenberg,
            "Underwood":self.Underwood,
            "NF":self.NF,
            "GHR_M1":self.GHR_M1,
            "GHR_M2":self.GHR_M2,
            "GHR_M3":self.GHR_M3,
            "KK":self.KK,
            "Jayakrishnan":self.Jayakrishnan,
            "Van_Aerde":self.Van_Aerde,
            "MacNicholas":self.MacNicholas,
            "Wang_3PL":self.Wang_3PL,
            "Wang_4PL":self.Wang_4PL,
            "Wang_5PL":self.Wang_5PL,
            "Ni":self.Ni,
            "S3_joint_estimation":self.S3_joint_estimation,
            "Greenshields_joint_estimation":self.Greenshields_joint_estimation,
            "Greenberg_joint_estimation":self.Greenberg_joint_estimation,
            "Underwood_joint_estimation":self.Underwood_joint_estimation,
            "NF_joint_estimation":self.NF_joint_estimation,
            "GHR_M1_joint_estimation":self.GHR_M1_joint_estimation,
            "GHR_M2_joint_estimation":self.GHR_M2_joint_estimation,
            "GHR_M3_joint_estimation":self.GHR_M3_joint_estimation,
            "KK_joint_estimation":self.KK_joint_estimation,
            "Jayakrishnan_joint_estimation":self.Jayakrishnan_joint_estimation,
            "Van_Aerde_joint_estimation":self.Van_Aerde_joint_estimation,
            "MacNicholas_joint_estimation":self.MacNicholas_joint_estimation,
            "Wang_3PL_joint_estimation":self.Wang_3PL_joint_estimation,
            "Wang_4PL_joint_estimation":self.Wang_4PL_joint_estimation,
            "Wang_5PL_joint_estimation":self.Wang_5PL_joint_estimation,
            "Ni_joint_estimation":self.Ni_joint_estimation,
        }


class Theoretical_Value:

    def S3(self, beta, density):
        vf, kc, foc = beta
        theoretical_speed = vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        theoretical_flow = density*vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        return theoretical_speed, theoretical_flow

    def Greenshields(self, beta, density):
        vf, k_jam = beta
        theoretical_speed = vf*(1 - density/k_jam)
        theoretical_flow = density*vf*(1 - density/k_jam)
        return theoretical_speed, theoretical_flow

    def Greenberg(self, beta, density):
        vc, k_jam = beta
        theoretical_speed = vc*np.log(k_jam/density)
        theoretical_flow = density*vc*np.log(k_jam/density)
        return theoretical_speed, theoretical_flow

    def Underwood(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-1*density/kc)
        theoretical_flow = density*vf*np.exp(-1*density/kc)
        return theoretical_speed, theoretical_flow

    def NF(self, beta, density):
        vf, k_jam, lambda_NF = beta
        theoretical_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        theoretical_flow = density*vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        return theoretical_speed, theoretical_flow

    def GHR_M1(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-0.5*np.power(density/kc, 2))
        theoretical_flow = density*vf*np.exp(-0.5*np.power(density/kc, 2))
        return theoretical_speed, theoretical_flow

    def GHR_M2(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*(1 - np.power(density/k_jam, m))
        theoretical_flow = density*vf*(1 - np.power(density/k_jam, m))
        return theoretical_speed, theoretical_flow

    def GHR_M3(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        return theoretical_speed, theoretical_flow

    def KK(self, beta, density):
        vf, kc, c1, c2, c3 = beta
        theoretical_speed = vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        theoretical_flow = density*vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        return theoretical_speed, theoretical_flow

    def Jayakrishnan(self, beta, density):
        vf, v_jam, k_jam, m = beta
        theoretical_speed = v_jam + (vf - v_jam)*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*(v_jam + (vf - v_jam)*(np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m))
        return theoretical_speed, theoretical_flow

    def Van_Aerde(self, beta, speed):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        theoretical_density = 1/(c1 + c2/(vf-speed) + c3*speed)
        theoretical_flow = speed/(c1 + c2/(vf-speed) + c3*speed)
        return theoretical_density, theoretical_flow

    def MacNicholas(self, beta, density):
        vf, k_jam, m, c = beta
        theoretical_speed = vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        theoretical_flow = density*vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        return theoretical_speed, theoretical_flow

    def Wang_3PL(self, beta, density):
        vf, kc, theta = beta
        theoretical_speed = vf/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*vf/(1+np.exp((density-kc)/theta))
        return theoretical_speed, theoretical_flow

    def Wang_4PL(self, beta, density):
        vf, vb, kc, theta = beta
        theoretical_speed = vb + (vf-vb)/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*(vb + (vf-vb)/(1+np.exp((density-kc)/theta)))
        return theoretical_speed, theoretical_flow

    def Wang_5PL(self, beta, density):
        vf, vb, kc, theta1, theta2 = beta
        theoretical_speed = vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2)
        theoretical_flow = density*(vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2))
        return theoretical_speed, theoretical_flow

    def Ni(self, beta, speed):
        vf, gamma, tao, l = beta
        theoretical_density = 1/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        theoretical_flow = speed/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        return theoretical_density, theoretical_flow

    def S3_joint_estimation(self, beta, density):
        vf, kc, foc = beta
        theoretical_speed = vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        theoretical_flow = density*vf/np.power(1 + np.power((density/kc), foc), 2/foc)
        return theoretical_speed, theoretical_flow

    def Greenshields_joint_estimation(self, beta, density):
        vf, k_jam = beta
        theoretical_speed = vf*(1 - density/k_jam)
        theoretical_flow = density*vf*(1 - density/k_jam)
        return theoretical_speed, theoretical_flow

    def Greenberg_joint_estimation(self, beta, density):
        vc, k_jam = beta
        theoretical_speed = vc*np.log(k_jam/density)
        theoretical_flow = density*vc*np.log(k_jam/density)
        return theoretical_speed, theoretical_flow

    def Underwood_joint_estimation(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-1*density/kc)
        theoretical_flow = density*vf*np.exp(-1*density/kc)
        return theoretical_speed, theoretical_flow

    def NF_joint_estimation(self, beta, density):
        vf, k_jam, lambda_NF = beta
        theoretical_speed = vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        theoretical_flow = density*vf*(1-np.exp(-1*lambda_NF/vf*(1/density - 1/k_jam)))
        return theoretical_speed, theoretical_flow

    def GHR_M1_joint_estimation(self, beta, density):
        vf, kc = beta
        theoretical_speed = vf*np.exp(-0.5*np.power(density/kc, 2))
        theoretical_flow = density*vf*np.exp(-0.5*np.power(density/kc, 2))
        return theoretical_speed, theoretical_flow

    def GHR_M2_joint_estimation(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*(1 - np.power(density/k_jam, m))
        theoretical_flow = density*vf*(1 - np.power(density/k_jam, m))
        return theoretical_speed, theoretical_flow

    def GHR_M3_joint_estimation(self, beta, density):
        vf, k_jam, m = beta
        theoretical_speed = vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*vf*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        return theoretical_speed, theoretical_flow

    def KK_joint_estimation(self, beta, density):
        vf, kc, c1, c2, c3 = beta
        theoretical_speed = vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        theoretical_flow = density*vf*(1/(1 + np.exp((density/kc - c1)/c2)) - c3)
        return theoretical_speed, theoretical_flow

    def Jayakrishnan_joint_estimation(self, beta, density):
        vf, v_jam, k_jam, m = beta
        theoretical_speed = v_jam + (vf - v_jam)*np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m
        theoretical_flow = density*(v_jam + (vf - v_jam)*(np.sign(1-density/k_jam)*np.abs(1-density/k_jam)**m))
        return theoretical_speed, theoretical_flow

    def Van_Aerde_joint_estimation(self, beta, speed):
        vf, vc, k_jam, max_flow = beta
        c1 = vf/(k_jam*np.power(vc, 2))*(2*vc - vf)
        c2 = vf/(k_jam*np.power(vc, 2))*np.power(vf - vc, 2)
        c3 = 1/max_flow - vf/(k_jam*np.power(vc, 2))
        theoretical_density = 1/(c1 + c2/(vf-speed) + c3*speed)
        theoretical_flow = speed/(c1 + c2/(vf-speed) + c3*speed)
        return theoretical_density, theoretical_flow

    def MacNicholas_joint_estimation(self, beta, density):
        vf, k_jam, m, c = beta
        theoretical_speed = vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        theoretical_flow = density*vf*(np.power(k_jam, m) - np.power(density, m))/(np.power(k_jam, m) + c*np.power(density, m))
        return theoretical_speed, theoretical_flow

    def Wang_3PL_joint_estimation(self, beta, density):
        vf, kc, theta = beta
        theoretical_speed = vf/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*vf/(1+np.exp((density-kc)/theta))
        return theoretical_speed, theoretical_flow

    def Wang_4PL_joint_estimation(self, beta, density):
        vf, vb, kc, theta = beta
        theoretical_speed = vb + (vf-vb)/(1+np.exp((density-kc)/theta))
        theoretical_flow = density*(vb + (vf-vb)/(1+np.exp((density-kc)/theta)))
        return theoretical_speed, theoretical_flow

    def Wang_5PL_joint_estimation(self, beta, density):
        vf, vb, kc, theta1, theta2 = beta
        theoretical_speed = vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2)
        theoretical_flow = density*(vb + (vf-vb)/np.power(1+np.exp((density-kc)/theta1), theta2))
        return theoretical_speed, theoretical_flow

    def Ni_joint_estimation(self, beta, speed):
        vf, gamma, tao, l = beta
        theoretical_density = 1/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        theoretical_flow = speed/((gamma*np.power(speed,2)+tao*speed+l)*(1-np.log(1-speed/vf)))
        return theoretical_density, theoretical_flow

    def theoretical_value_func_dict(self):
        return {
            "S3":self.S3,
            "Greenshields":self.Greenshields,
            "Greenberg":self.Greenberg,
            "Underwood":self.Underwood,
            "NF":self.NF,
            "GHR_M1":self.GHR_M1,
            "GHR_M2":self.GHR_M2,
            "GHR_M3":self.GHR_M3,
            "KK":self.KK,
            "Jayakrishnan":self.Jayakrishnan,
            "Van_Aerde":self.Van_Aerde,
            "MacNicholas":self.MacNicholas,
            "Wang_3PL":self.Wang_3PL,
            "Wang_4PL":self.Wang_4PL,
            "Wang_5PL":self.Wang_5PL,
            "Ni":self.Ni,
            "S3_joint_estimation":self.S3_joint_estimation,
            "Greenshields_joint_estimation":self.Greenshields_joint_estimation,
            "Greenberg_joint_estimation":self.Greenberg_joint_estimation,
            "Underwood_joint_estimation":self.Underwood_joint_estimation,
            "NF_joint_estimation":self.NF_joint_estimation,
            "GHR_M1_joint_estimation":self.GHR_M1_joint_estimation,
            "GHR_M2_joint_estimation":self.GHR_M2_joint_estimation,
            "GHR_M3_joint_estimation":self.GHR_M3_joint_estimation,
            "KK_joint_estimation":self.KK_joint_estimation,
            "Jayakrishnan_joint_estimation":self.Jayakrishnan_joint_estimation,
            "Van_Aerde_joint_estimation":self.Van_Aerde_joint_estimation,
            "MacNicholas_joint_estimation":self.MacNicholas_joint_estimation,
            "Wang_3PL_joint_estimation":self.Wang_3PL_joint_estimation,
            "Wang_4PL_joint_estimation":self.Wang_4PL_joint_estimation,
            "Wang_5PL_joint_estimation":self.Wang_5PL_joint_estimation,
            "Ni_joint_estimation":self.Ni_joint_estimation,
        }


class GetMetrics:

    def __init__(self, observed_flow: np.array, observed_density: np.array, observed_speed: np.array, estimated_flow: np.array, estimated_speed: np.array):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        self.estimated_flow = estimated_flow
        self.estimated_speed = estimated_speed

    def RMSE_Overall(self) -> list:
        rmse_speed = mean_squared_error(self.observed_speed, self.estimated_speed, squared=False)
        rmse_flow = mean_squared_error(self.observed_flow, self.estimated_flow, squared=False)
        r2_speed = r2_score(self.observed_speed, self.estimated_speed)
        r2_flow = r2_score(self.observed_flow, self.estimated_flow)
        return [rmse_speed, rmse_flow, r2_speed, r2_flow]

    def RMSE_Small_Range(self, interval: int = 10) -> list:
        rmse_speed_small_range = []
        rmse_flow_small_range = []
        density_max_value = min(math.ceil(max(self.observed_density) / 10), 10)
        for i in range(density_max_value):
            temp_index = np.where((self.observed_density >= interval*i) & (self.observed_density < interval*(i+1)))
            observed_speed_i = self.observed_speed[temp_index]
            estimated_speed_i = self.estimated_speed[temp_index]
            observed_flow_i = self.observed_flow[temp_index]
            estimated_flow_i = self.estimated_flow[temp_index]
            try:
                rmse_speed_small_range.append(mean_squared_error(observed_speed_i, estimated_speed_i, squared=False))
            except Exception:
                rmse_speed_small_range.append(0)

            try:
                rmse_flow_small_range.append(mean_squared_error(observed_flow_i, estimated_flow_i, squared=False))
            except Exception:
                rmse_flow_small_range.append(0)
        observed_speed_last = self.observed_speed[np.where((self.observed_density >= density_max_value))]
        estimated_speed_last = self.estimated_speed[np.where((self.observed_density >= density_max_value))]
        observed_flow_last = self.observed_flow[np.where((self.observed_density >= density_max_value))]
        estimated_flow_last = self.estimated_flow[np.where((self.observed_density >= density_max_value))]

        try:
            rmse_speed_small_range.append(mean_squared_error(observed_speed_last, estimated_speed_last, squared=False))
        except Exception:
            rmse_speed_small_range.append(0)

        try:
            rmse_flow_small_range.append(mean_squared_error(observed_flow_last, estimated_flow_last, squared=False))
        except Exception:
            rmse_flow_small_range.append(0)

        return rmse_speed_small_range, rmse_flow_small_range
