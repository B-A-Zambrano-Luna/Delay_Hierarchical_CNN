# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 17:41:39 2022

@author: LENOVO
"""
from skimage.util.shape import view_as_blocks
import numpy as np
import image2test
from Q_p_delay_CNN import p_adic_delay_CNN
from matplotlib import pyplot as plt
# from p_adic_aux_function import \
#     vectorize_function,\
#     p_adic_Convolution_matrix


def f(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


def DCNN_2_image(image, J, A, B, Z,
                 X_0, t, delta_t, Z_k,
                 nonlineality=f, delay=-1, L=1,
                 split_image=True,
                 screem_shot=True,
                 reduce=False,
                 without_label=True):

    if split_image:
        p = Z_k.get_prime()
        K = Z_k.get_radio()
        size = p**(int(K/2))
        sub_image = view_as_blocks(image, (size, size))
        # len_times = int(t/delta_t)
        record_aux = dict()
        # record_time = dict()
        hight = sub_image.shape[0]
        wide = sub_image.shape[1]
        scheme_image = image2test.imate2test(np.zeros([size, size]), Z_k,
                                             reduction=False)
        scheme_image.fit()
        for i in range(0, hight):
            for j in range(0, wide):
                image = sub_image[i, j]
                image_12test = image2test.imate2test(image, Z_k,
                                                     reduction=False)
                image_12test.fit()
                # scheme_image.image = image
                # U = scheme_image.get_values()
                U = image_12test.get_values()

                # Input

                solver = p_adic_delay_CNN()
                solver.solution(J, A, B, U, Z,
                                X_0, t, delta_t, Z_k,
                                nonlineality=nonlineality,
                                delay=delay, L=L)

                record_aux[(i, j)] = solver.ode_result  # row->time
        record = dict()
        scheme_image = image2test.imate2test(np.zeros([size, size]), Z_k,
                                             reduction=False)
        scheme_image.fit()
        if screem_shot:
            times_position = [int(t0/delta_t) for t0 in range(int(t)+1)]
        elif not screem_shot:
            times_position = [t for t in range(int(t/delta_t)+1)]
        for t in times_position:
            # time_output_aux = []
            for i in range(hight):
                for j in range(wide):
                    sub_image[i, j] = scheme_image.\
                        inverse_transform(record_aux[(i, j)][t][:])
            image_0 = [np.concatenate(sub_image[i],
                                      axis=1)
                       for i in range(hight)]
            record[t] = np.concatenate(image_0, axis=0)

        """ Image plot X"""
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in times_position:
            plt.imshow(record[t], cmap='gray')
            plt.title("X at Time " + str(round(t*delta_t, decimal_place)))
            if without_label:
                plt.xticks([])
                plt.yticks([])
            plt.show()
        """ Image plot Y"""
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in times_position:
            plt.imshow(nonlineality(record[t]), cmap='gray')
            plt.title("Y at Time " + str(round(t*delta_t, decimal_place)))
            if without_label:
                plt.xticks([])
                plt.yticks([])
            plt.show()

    elif not split_image:
        image_12test = image2test.imate2test(image, Z_k,
                                             reduction=reduce)
        image_12test.fit()
        U = image_12test.get_values()
        # Input
        solver = p_adic_delay_CNN()
        solver.solution(J, A, B, U, Z,
                        X_0, t, delta_t, Z_k,
                        nonlineality=nonlineality,
                        delay=delay, L=L)

        A1 = solver.ode_result  # row->time
        if screem_shot:
            times_position = [int(t0/delta_t) for t0 in range(int(t)+1)]
        elif not screem_shot:
            times_position = [t for t in range(int(t/delta_t)+1)]

        """ Image plot X"""
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in times_position:
            image_reves = image_12test.\
                inverse_transform(A1[t])
            plt.imshow(image_reves, cmap='gray')
            plt.title("X at Time " + str(round(t*delta_t, decimal_place)))
            if without_label:
                plt.xticks([])
                plt.yticks([])
            plt.show()

        """ Image plot Y"""
        if type(delta_t) == int:
            decimal_place = 1
        else:
            decimal_place = len(str(delta_t).split(".")[1])
        for t in times_position:
            image_reves = image_12test.\
                inverse_transform(A1[t])
            plt.imshow(nonlineality(image_reves), cmap='gray')
            plt.title("Y at Time " + str(round(t*delta_t, decimal_place)))
            if without_label:
                plt.xticks([])
                plt.yticks([])
            plt.show()
    return record
