# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:40:23 2022

@author: LENOVO
"""
import numpy as np
import seaborn as sns
import joblib
from matplotlib import pyplot as plt
from p_adic_aux_function import \
    vectorize_function,\
    p_adic_Convolution_matrix
import Q_p
import pandas as pd
from Q_p_as_fractal import Christiakov_emmending


def nonlineality(x):
    """
    Parameters
    ----------
    x : TYPE 2d or 1d-Numpy array
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return 0.5*(abs(x+1) - abs(x-1))


class p_adic_delay_CNN(object):

    def __init__(self):
        """
        Returns
        -------
        None.

        """
        self.nonlineality = nonlineality
        self.L = 1
        self.ode_result = []
        self.J_matrix = []
        self.A_matrix = []
        self.B_matrix = []
        self.U_vect = []
        self.Z_vect = []
        self.X_0_vect = []
        self.time = 0
        self.delta_t = 0
        self.delay = 0

    def get_X_0(self):
        return self.X_0_vect.copy()

    def dt(self, t, X, delay_X):
        """


        Parameters
        ----------
        t : TYPE float
            DESCRIPTION.
        X : TYPE p^k-array
            DESCRIPTION. State X
        delay_X : TYPE p^k-array
            DESCRIPTION. delay State X    

        Returns
        -------
        dX : TYPE p^K-array
            DESCRIPTION.
        """
        L = self.L
        f = self.nonlineality
        JX = np.matmul(self.J_matrix, X)

        AX = np.matmul(self.A_matrix, f(delay_X))

        BU = np.matmul(self.B_matrix, self.U_vect)

        dX = -L*X + JX + AX + BU + self.Z_vect

        return dX

    def save(self, name=""):
        joblib.dump(self, "delay_CNN_solver_"+name+".pkl")

    def solution(self, J, A, B, U, Z,
                 X_0, t, delta_t, Z_k,
                 nonlineality=nonlineality,
                 delay=-1, L=1):
        """
        We use Euler forward algorithm since we are working
        with evolution equations.

        h should be positive
        r should be negative
        Parameters
        ----------
        J : TYPE
            DESCRIPTION.
        A : TYPE
            DESCRIPTION.
        B : TYPE
            DESCRIPTION.
        U : TYPE
            DESCRIPTION.
        Z : TYPE
            DESCRIPTION.
        X_0 : TYPE callable function
            DESCRIPTION. If delay>0 X_0(x,t) is a 
            p-adic x real function other case  X_0 = X_0(x)
        t : TYPE
            DESCRIPTION.
        delta_t : TYPE
            DESCRIPTION.
        Z_k : TYPE
            DESCRIPTION.
        nonlineality : TYPE, optional
            DESCRIPTION. The default is nonlineality.
        L : TYPE, optional
            DESCRIPTION. The default is 1.
        delay : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        None.

        """

        self.Z_k = Z_k
        self.nonlineality = nonlineality
        self.L = L
        self.time = t
        self.delta_t = delta_t
        self.delay = delay
        p = Z_k.get_prime()
        k = Z_k.get_radio()
        if J == []:
            self.J_matrix = p_adic_Convolution_matrix(J, Z_k)
        if A == []:
            self.A_matrix = p_adic_Convolution_matrix(A, Z_k)
        self.U_vect = vectorize_function(U, Z_k)
        self.Z_vect = vectorize_function(Z, Z_k)
        if self.delay >= 0:
            self.X_0_vect = vectorize_function(X_0, Z_k)
            self.ode_result = np.array([self.X_0_vect])
        else:
            self.X_0_vect = np.empty([1, p**k])
            s = self.delay
            while s <= 0:
                def X_0_aux(x):
                    return X_0(x, s)
                X_0_vect = np.array([vectorize_function(X_0_aux, Z_k)])
                self.X_0_vect = np.concatenate(
                    [self.X_0_vect, X_0_vect],
                    axis=0)
                s += self.delta_t
            self.X_0_vect = self.X_0_vect[1:]
            self.ode_result = self.X_0_vect[1:]
        d_t = 0
        delay_step = 0
        with_delay = self.delay < 0
        while d_t <= t:
            if with_delay:

                dX = self.dt(d_t, self.ode_result[-1],
                             self.ode_result[delay_step])

                delay_step += 1
            else:
                dX = self.dt(d_t, self.ode_result[-1],
                             self.ode_result[-1])
            # print(dX, "dX")
            dtX = np.array([self.ode_result[-1]
                            + delta_t*dX])

            self.ode_result = np.concatenate(
                [self.ode_result, dtX],
                axis=0)
            d_t += delta_t

    def get_result(self):
        return np.rot90(self.ode_result, k=1, axes=(0, 1))

    def plot(self, function="state", ylabels=True, xlabels=True,
             with_title=True):
        if function == "state":
            matrix = np.rot90(self.ode_result, k=1, axes=(0, 1))
        elif function == "output":
            matrix = self.nonlineality(
                np.rot90(self.ode_result, k=1, axes=(0, 1)))
        elif function == "A":
            matrix = self.A_matrix
        elif function == "J":
            matrix = self.J_matrix
        elif function == "B":
            matrix = self.B_matrix
        elif function == "intital":
            if self.r < 0:
                matrix = self.X_0_vect
            else:
                matrix = np.array([self.X_0_vect])
        elif function == "U":
            matrix = np.array([self.U_vect])
        elif function == "Z":
            matrix = np.array([self.Z_vect])

        k = self.Z_k.get_radio()
        p = self.Z_k.get_prime()

        if function == "state" or function == "output":
            ax = sns.heatmap(matrix)
            ax.set(xlabel="Time",
                   ylabel="p-adic integers")
            if with_title:
                plt.title("Function " + function)
            else:
                plt.title("")
            x_ticks = ax.xaxis.get_ticklocs()
            if xlabels:
                t_step = int(self.time/self.delta_t)
                step = round(t_step/len(x_ticks))
                ax.xaxis.set_ticklabels([round(t*step*self.delta_t, 2)
                                         for t in range(len(x_ticks))])
            else:
                ax.xaxis.set_ticklabels([])

            if k <= 2 and ylabels:
                Z_k_list = list(self.Z_k)[::-1]
                ax.yaxis.set_ticklabels(Z_k_list,
                                        rotation=0)
            else:
                ax.yaxis.set_ticklabels([], rotation=0)
        elif function in ["U", "Z", "initial"]:
            fig, ax = plt.subplots(figsize=(12, 2))
            ax = sns.heatmap(matrix, ax=ax)
            if with_title:
                plt.title("Function " + function)
            else:
                plt.title("")
            if function == "initial" and self.delay < 0:
                ax.set(xlabel="p-adic integers",
                       ylabel="time")
            else:
                ax.set(xlabel="p-adic integers",
                       ylabel="")
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])
            if xlabels:
                Z_k_list = list(self.Z_k)
                ax.xaxis.set_ticklabels(Z_k_list,
                                        rotation=0)
            else:
                ax.xaxis.set_ticklabels([])
        elif function in ["J", "A", "B"]:
            Z_k_list = list(self.Z_k)
            ax = sns.heatmap(matrix)
            ax.set(xlabel="p-adic integers",
                   ylabel="p-adic integers")
            if with_title:
                plt.title("Kernel " + function)
            else:
                plt.title("")
            if xlabels and ylabels:
                x_ticks = ax.xaxis.get_ticklocs()
                y_ticks = ax.yaxis.get_ticklocs()
                step = round(len(self.Z_k)/len(x_ticks))
                ax.xaxis.set_ticklabels([Z_k_list[a*step]
                                         for a in range(len(x_ticks))],
                                        rotation=90)
                ax.yaxis.set_ticklabels([Z_k_list[a*step]
                                         for a in range(len(y_ticks))],
                                        rotation=0)
            elif not xlabels and ylabels:
                ax.xaxis.set_ticklabels([])
                y_ticks = ax.yaxis.get_ticklocs()
                step = round(len(self.Z_k)/len(y_ticks))
                ax.yaxis.set_ticklabels([Z_k_list[a*step]
                                         for a in range(len(y_ticks))],
                                        rotation=0)
            elif not ylabels and xlabels:
                x_ticks = ax.xaxis.get_ticklocs()
                step = round(len(self.Z_k)/len(x_ticks))
                ax.xaxis.set_ticklabels([Z_k_list[a*step]
                                         for a in range(len(x_ticks))],
                                        rotation=0)
                ax.yaxis.set_ticklabels([])
            else:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

    def plot_fract(self, m, s,
                   function="state",
                   labels=True,
                   size_points=0.5,
                   screen_shot=True,
                   with_title=True):
        x_position, y_position = Christiakov_emmending(self.Z_k, m, s)
        if function == "state":
            matrix = np.rot90(self.ode_result, k=1, axes=(0, 1))
        elif function == "output":
            matrix = self.nonlineality(
                np.rot90(self.ode_result, k=1, axes=(0, 1)))
        elif function == "A":
            matrix = self.A_matrix
        elif function == "J":
            matrix = self.J_matrix
        elif function == "B":
            matrix = self.B_matrix
        elif function == "intital":
            if self.r < 0:
                matrix = self.X_0_vect
            else:
                matrix = np.array([self.X_0_vect])
        elif function == "U":
            matrix = np.array([self.U_vect])
        elif function == "Z":
            matrix = np.array([self.Z_vect])
        if function == "state" or function == "output":
            maxValue = matrix.max()
            minValue = matrix.min()
            if screen_shot:
                times = [int(t/self.delta_t) for t in range(int(self.time)+1)]
                for t in times:
                    im = plt.scatter(x_position,
                                     y_position,
                                     c=matrix[:, t][::-1],
                                     s=size_points,
                                     vmax=maxValue,
                                     vmin=minValue)
                    plt.colorbar()

                    if with_title:
                        plt.title("Layer "+function +
                                  " time " + str(int(t*self.delta_t)))
                    elif not with_title:
                        plt.title("")
                    if not labels:
                        plt.xticks([])
                        plt.yticks([])
                    plt.show()
            elif not screen_shot:
                times = matrix.shape[1]
                for t in range(times):
                    im = plt.scatter(x_position,
                                     y_position,
                                     c=matrix[:, t][::-1],
                                     s=size_points,
                                     vmax=maxValue,
                                     vmin=minValue)
                    if labels:
                        plt.title("Layer "+function +
                                  " time " +
                                  str(round(t*self.delta_t, 2)))
                    elif not labels:
                        plt.title("")
                        plt.xticks([])
                        plt.yticks([])
                    plt.colorbar()
                    plt.show()

        elif function in ["A", "B", "J"]:
            im = plt.scatter(x_position,
                             y_position,
                             c=matrix[0][:],
                             s=size_points)
            plt.colorbar()
            if with_title:
                plt.title("Kernel "+function)
            elif not with_title:
                plt.title("")
            if not labels:
                plt.xticks([])
                plt.yticks([])
            plt.show()
        elif function in ["U", "Z"]:
            im = plt.scatter(x_position,
                             y_position,
                             c=matrix,
                             s=size_points)
            plt.colorbar()
            if with_title:
                plt.title("Function "+function)
            elif not with_title:
                plt.title("")
            if not labels:
                plt.xticks([])
                plt.yticks([])
            plt.show()
