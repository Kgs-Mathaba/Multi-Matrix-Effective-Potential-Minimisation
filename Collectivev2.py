import pandas as pd
import numpy as np
from scipy import linalg
from numpy.linalg import multi_dot
import datetime
import csv
import pathlib
import os


def initialize_loops(omega_dim, num_loops):
    """
    This function intializes loops, omega, little_omega and LnJ with zeros

    Argument:
    omega_dim -- size of omega matrix
    num_loops -- number of loops

    Returns:
    loop -- initialized complex array of shape (num_loops,)
    omega --  initialized 2d complex array of shape (omega_dim,omega_dim)
    little_omega -- initialized complex array of shape (omega_dim,)
    LnJ -- initialized complex array of shape (omega_dim,)
    """

    loop = np.zeros((num_loops + 1), dtype=complex)
    omega = np.zeros((omega_dim, omega_dim), dtype=complex)
    little_omega = np.zeros((omega_dim), dtype=complex)
    LnJ = np.zeros((omega_dim), dtype=complex)

    return loop, omega, little_omega, LnJ


def compare(Loop1, Loop2):
    """
    Compare two loops up to cyclic permutations
    """
    Length1 = len(Loop1)
    Length2 = len(Loop2)

    Same = False
    if Length2 == 0:
        Same = True
        return Same

    # Create warning if loops passed with different lengths #print ("lengths", Length1, Length2)

    Same = False
    for i in range(Length2):
        if np.array_equal(Loop1, np.roll(Loop2, i)):
            Same = True
            break

    return Same


def cyclic_generator(lp):
    """

    Input:   lp is a (1,len(lp)) ndarray
    Output:  is an array containing the first of the lexicographically sorted set of cyclic permutations of lp
             - example: [2,1,2,1] --> [1,2,1,2]
    """

    cyclic_perm = np.array([[0] * len(lp)] * len(lp))

    for i in range(len(lp)):
        cyclic_perm[i] = np.roll(lp, i)

    first_lexi_array = np.array(sorted([tuple(row) for row in cyclic_perm]))[0:1, :]

    return first_lexi_array


def effect_pot(
    x_N_2,
    omega_dim,
    num_loops,
    litte_omega_dim,
    nt_hooft,
    m_coupling,
    gYM_coupling,
    loop_list,
    non_zero,
    non_zero_lo,
    nonzero_index,
    nonzero_y,
    nonzero_z,
    nonzero_loop1_index,
    nonzero_loop2_index,
):

    loop, omega, little_omega, LnJ = initialize_loops(omega_dim, num_loops)

    # print()
    # print("Function call")
    # func_start= timer()

    matrices_array = np.zeros((2, nt_hooft, nt_hooft), dtype=complex)
    matrices_array[0] = np.diag(x_N_2[0:nt_hooft])
    temp_matrix_2 = np.reshape(
        x_N_2[nt_hooft : nt_hooft ** 2 + nt_hooft], (nt_hooft, nt_hooft)
    )
    real_matrix_2 = np.triu(temp_matrix_2)
    real_matrix_2 = real_matrix_2 + real_matrix_2.T - np.diag(np.diag(temp_matrix_2))
    imag_matrix_2 = np.tril(temp_matrix_2)
    imag_matrix_2 = imag_matrix_2 - imag_matrix_2.T
    matrices_array[1] = real_matrix_2 + imag_matrix_2 * 1.0j

    for i in range(1, num_loops + 1):
        loop_matrix = matrices_array[loop_list[i][0] - 1]
        for k in range(1, len(loop_list[i])):
            if loop_list[i][k] == 1:
                loop_matrix = (
                    loop_matrix * x_N_2[0:nt_hooft]
                )  # [k]#matrices_list[loop_list[i][k]-1]
            else:
                loop_matrix = np.dot(
                    loop_matrix, matrices_array[1]
                )  # [loop_list[i][k]-1]
        loop[i] = np.trace(loop_matrix) / (
            np.power(nt_hooft, (len(loop_list[i]) + 2) / 2)
        )  # v3
    loop[0] = 1.0

    # omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range(omega_dim):
        for j in range(omega_dim):
            omega[i, j] = 0.0
            for indx in range(non_zero[i + 1, j + 1]):
                k = nonzero_index[i + 1, j + 1, indx]
                y = nonzero_y[i + 1, j + 1, indx]
                omega[i, j] += y * loop[k]

    # little_omega=np.zeros ((omega_dim),dtype=complex)
    for i in range(omega_dim):
        little_omega[i] = 0.0
        for indx in range(non_zero_lo[i + 1]):
            j = nonzero_loop1_index[i + 1, indx]
            k = nonzero_loop2_index[i + 1, indx]
            z = nonzero_z[i + 1, indx]
            little_omega[i] += z * loop[j] * loop[k]
    # little_omega+=- omega[:,2]/2.0-omega[:,4]/2.0 + c_coupling*omega[:,3]-g3_coupling*omega[:,5]/3.0-g3_coupling*omega[:,8]/3.0-g4_coupling*omega[:,9]-g4_coupling*omega[:,14]

    LnJ = np.linalg.solve(omega, little_omega)

    large_n_energy = (
        np.dot(little_omega, LnJ) / 8.0
        + np.power(m_coupling, 2) * loop[3] / 2.0
        + np.power(m_coupling, 2) * loop[5] / 2.0
        + 2 * np.power(gYM_coupling, 2) * loop[12]
        - 2 * np.power(gYM_coupling, 2) * loop[13]
    )

    # print("Large N Energy :")
    print(large_n_energy)

    return large_n_energy.real


def effect_pot_grad(
    x_N_2,
    omega_dim,
    num_loops,
    litte_omega_dim,
    nt_hooft,
    m_coupling,
    gYM_coupling,
    loop_list,
    non_zero,
    non_zero_lo,
    nonzero_index,
    nonzero_y,
    nonzero_z,
    nonzero_loop1_index,
    nonzero_loop2_index,
):

    loop, omega, little_omega, LnJ = initialize_loops(omega_dim, num_loops)

    # print()
    # print("Gradient function call")
    # grad_start= timer()

    matrices_array = np.zeros((2, nt_hooft, nt_hooft), dtype=complex)
    matrices_array[0] = np.diag(x_N_2[0:nt_hooft])
    temp_matrix_2 = np.reshape(
        x_N_2[nt_hooft : nt_hooft ** 2 + nt_hooft], (nt_hooft, nt_hooft)
    )
    real_matrix_2 = np.triu(temp_matrix_2)
    real_matrix_2 = real_matrix_2 + real_matrix_2.T - np.diag(np.diag(temp_matrix_2))
    imag_matrix_2 = np.tril(temp_matrix_2)
    imag_matrix_2 = imag_matrix_2 - imag_matrix_2.T
    matrices_array[1] = real_matrix_2 + imag_matrix_2 * 1.0j

    # Loops calculated with derivatives
    loop[0] = 1.0
    loop[1] = np.sum(x_N_2[0:nt_hooft]) / (np.power(nt_hooft, 1.5))
    loop[2] = np.trace(matrices_array[1]) / (np.power(nt_hooft, 1.5))

    grad_array_1 = np.zeros((num_loops + 1, nt_hooft), dtype=complex)
    grad_array_2 = np.zeros((num_loops + 1, nt_hooft, nt_hooft), dtype=complex)
    grad_list_2 = np.zeros((num_loops + 1, nt_hooft * nt_hooft), dtype=complex)
    # this assumes first 3 loops are [] [1] [2]
    grad_array_1[1] = np.ones(nt_hooft)
    grad_array_2[2] = np.identity(nt_hooft)
    grad_list_2[2] = (grad_array_2[2]).flatten()

    for i in range(3, num_loops + 1):  # assumes first 3 loops will alwyas be the same
        for k in range(len(loop_list[i])):
            loop_left = loop_list[i][0:k]
            loop_right = loop_list[i][k + 1 : len(loop_list[i])]
            loop_grad = np.concatenate((loop_right, loop_left))
            grad_len = len(loop_grad)
            grad_matrix = matrices_array[loop_grad[0] - 1]

            if grad_len > 1:
                for l in range(1, grad_len):
                    if loop_grad[l] == 1:
                        grad_matrix = (
                            grad_matrix * x_N_2[0:nt_hooft]
                        )  # [k]#matrices_list[loop_list[i][k]-1]
                    else:
                        grad_matrix = np.dot(grad_matrix, matrices_array[1])
            if loop_list[i][k] == 1:
                grad_array_1[i] += np.diagonal(grad_matrix.real)
            else:
                grad_array_2[i] += grad_matrix

        last_index = len(loop_list[i]) - 1
        if loop_list[i][last_index] == 1:
            loop_matrix = grad_matrix * x_N_2[0:nt_hooft]
        else:
            loop_matrix = np.dot(grad_matrix, matrices_array[1])  # [

        loop[i] = np.trace(loop_matrix) / (
            np.power(nt_hooft, (len(loop_list[i]) + 2) / 2)
        )
        upper_matrix_2 = np.triu((grad_array_2[i] + grad_array_2[i].T)) - np.diag(
            np.diag(grad_array_2[i])
        )
        lower_matrix_2 = -1.0j * (np.tril((grad_array_2[i] - grad_array_2[i].T)))
        grad_array_2[i] = upper_matrix_2 + lower_matrix_2
        grad_list_2[i] = (grad_array_2[i]).flatten()

    # omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range(omega_dim):
        for j in range(omega_dim):
            omega[i, j] = 0.0
            for indx in range(non_zero[i + 1, j + 1]):
                k = nonzero_index[i + 1, j + 1, indx]
                y = nonzero_y[i + 1, j + 1, indx]
                omega[i, j] += y * loop[k]

    # little_omega=np.zeros ((omega_dim),dtype=complex)
    for i in range(omega_dim):
        little_omega[i] = 0.0
        for indx in range(non_zero_lo[i + 1]):
            j = nonzero_loop1_index[i + 1, indx]
            k = nonzero_loop2_index[i + 1, indx]
            z = nonzero_z[i + 1, indx]
            little_omega[i] += z * loop[j] * loop[k]
    # little_omega+=- omega[:,2]/2.0-omega[:,4]/2.0 + c_coupling*omega[:,3]-g3_coupling*omega[:,5]/3.0-g3_coupling*omega[:,8]/3.0-g4_coupling*omega[:,9]-g4_coupling*omega[:,14]

    LnJ = np.linalg.solve(omega, little_omega)

    grad_1 = np.zeros((nt_hooft), dtype=complex)  # dtype=np.float64)

    for i in range(omega_dim):
        for j in range(omega_dim):
            for indx in range(non_zero[i + 1, j + 1]):
                k = nonzero_index[i + 1, j + 1, indx]
                y = nonzero_y[i + 1, j + 1, indx]
                grad_1 += (
                    -LnJ[i]
                    * y
                    * LnJ[j]
                    * grad_array_1[k]
                    / (np.power(nt_hooft, (len(loop_list[k]) + 2) / 2))
                )

    for i in range(omega_dim):
        for indx in range(non_zero_lo[i + 1]):
            j = nonzero_loop1_index[i + 1, indx]
            k = nonzero_loop2_index[i + 1, indx]
            z = nonzero_z[i + 1, indx]
            grad_1 += (
                2
                * LnJ[i]
                * z
                * (
                    grad_array_1[j]
                    * loop[k]
                    / (np.power(nt_hooft, (len(loop_list[j]) + 2) / 2))
                    + loop[j]
                    * grad_array_1[k]
                    / (np.power(nt_hooft, (len(loop_list[k]) + 2) / 2))
                )
            )

    #    for i in range (omega_dim):
    #        for indx in range(non_zero[i+1,3]):
    #            k=nonzero_index[i+1,3,indx]
    #            y=nonzero_y[i+1,3,indx]
    #            grad_1+=(-2*LnJ[i]*(y/2.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,5]):
    #            k=nonzero_index[i+1,5,indx]
    #            y=nonzero_y[i+1,5,indx]
    #            grad_1+=(-2*LnJ[i]*(y/2.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,4]):
    #            k=nonzero_index[i+1,4,indx]
    #            y=nonzero_y[i+1,4,indx]
    #            grad_1+=(-2*LnJ[i]*(- c_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,6]):
    #            k=nonzero_index[i+1,6,indx]
    #            y=nonzero_y[i+1,6,indx]
    #            grad_1+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,9]):
    #            k=nonzero_index[i+1,9,indx]
    #            y=nonzero_y[i+1,9,indx]
    #            grad_1+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,10]):
    #            k=nonzero_index[i+1,10,indx]
    #            y=nonzero_y[i+1,10,indx]
    #            grad_1+=(-2*LnJ[i]*(g4_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,15]):
    #            k=nonzero_index[i+1,15,indx]
    #            y=nonzero_y[i+1,15,indx]
    #            grad_1+=(-2*LnJ[i]*(g4_coupling*y)*grad_array_1[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    grad_1 = grad_1 / 8.0
    grad_1 += (
        np.power(m_coupling, 2)
        * grad_array_1[3]
        / (2 * np.power(nt_hooft, (len(loop_list[3]) + 2) / 2))
    )
    grad_1 += (
        np.power(m_coupling, 2)
        * grad_array_1[5]
        / (2 * np.power(nt_hooft, (len(loop_list[5]) + 2) / 2))
    )
    # grad_1+= c_coupling *(grad_array_1[4]/ (np.power(nt_hooft, (len(loop_list[4])+2)/2)))
    # grad_1+= g3_coupling*(grad_array_1[6]/ (3*np.power(nt_hooft, (len(loop_list[6])+2)/2)))
    # grad_1+= g3_coupling*(grad_array_1[9]/ (3*np.power(nt_hooft, (len(loop_list[9])+2)/2)))
    grad_1 += (
        2
        * np.power(gYM_coupling, 2)
        * grad_array_1[12]
        / (np.power(nt_hooft, (len(loop_list[12]) + 2) / 2))
    )
    grad_1 -= (
        2
        * np.power(gYM_coupling, 2)
        * grad_array_1[13]
        / (np.power(nt_hooft, (len(loop_list[13]) + 2) / 2))
    )

    grad_2 = np.zeros((nt_hooft * nt_hooft), dtype=complex)

    for i in range(omega_dim):
        for j in range(omega_dim):
            for indx in range(non_zero[i + 1, j + 1]):
                k = nonzero_index[i + 1, j + 1, indx]
                y = nonzero_y[i + 1, j + 1, indx]
                grad_2 += (
                    -LnJ[i]
                    * y
                    * LnJ[j]
                    * grad_list_2[k]
                    / (np.power(nt_hooft, (len(loop_list[k]) + 2) / 2))
                )

    for i in range(omega_dim):
        for indx in range(non_zero_lo[i + 1]):
            j = nonzero_loop1_index[i + 1, indx]
            k = nonzero_loop2_index[i + 1, indx]
            z = nonzero_z[i + 1, indx]
            grad_2 += (
                2
                * LnJ[i]
                * z
                * (
                    grad_list_2[j]
                    * loop[k]
                    / (np.power(nt_hooft, (len(loop_list[j]) + 2) / 2))
                    + loop[j]
                    * grad_list_2[k]
                    / (np.power(nt_hooft, (len(loop_list[k]) + 2) / 2))
                )
            )

    #   for i in range (omega_dim):
    #       for indx in range(non_zero[i+1,3]):
    #            k=nonzero_index[i+1,3,indx]
    #            y=nonzero_y[i+1,3,indx]
    #            grad_2+=(-2*LnJ[i]*(y/2.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,5]):
    #            k=nonzero_index[i+1,5,indx]
    #            y=nonzero_y[i+1,5,indx]
    #            grad_2+=(-2*LnJ[i]*(y/2.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,4]):
    #            k=nonzero_index[i+1,4,indx]
    #            y=nonzero_y[i+1,4,indx]
    #            grad_2+=(-2*LnJ[i]*(- c_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,6]):
    #            k=nonzero_index[i+1,6,indx]
    #            y=nonzero_y[i+1,6,indx]
    #            grad_2+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,9]):
    #            k=nonzero_index[i+1,9,indx]
    #            y=nonzero_y[i+1,9,indx]
    #            grad_2+=(-2*LnJ[i]*(g3_coupling*y/3.0)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,10]):
    #            k=nonzero_index[i+1,10,indx]
    #            y=nonzero_y[i+1,10,indx]
    #            grad_2+=(-2*LnJ[i]*(g4_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    #        for indx in range(non_zero[i+1,15]):
    #            k=nonzero_index[i+1,15,indx]
    #            y=nonzero_y[i+1,15,indx]
    #            grad_2+=(-2*LnJ[i]*(g4_coupling*y)*grad_list_2[k]/ (np.power(nt_hooft, (len(loop_list[k])+2)/2)))
    grad_2 = grad_2 / 8.0
    grad_2 += (
        np.power(m_coupling, 2)
        * grad_list_2[3]
        / (2 * np.power(nt_hooft, (len(loop_list[3]) + 2) / 2))
    )
    grad_2 += (
        np.power(m_coupling, 2)
        * grad_list_2[5]
        / (2 * np.power(nt_hooft, (len(loop_list[5]) + 2) / 2))
    )
    # grad_2 += c_coupling *(grad_list_2[4]/ (np.power(nt_hooft, (len(loop_list[4])+2)/2)))
    # grad_2 += g3_coupling*(grad_list_2[6]/ (3*np.power(nt_hooft, (len(loop_list[6])+2)/2)))
    # grad_2 += g3_coupling*(grad_list_2[9]/ (3*np.power(nt_hooft, (len(loop_list[9])+2)/2)))
    grad_2 += (
        2
        * np.power(gYM_coupling, 2)
        * (grad_list_2[12] / (np.power(nt_hooft, (len(loop_list[12]) + 2) / 2)))
    )
    grad_2 -= (
        2
        * np.power(gYM_coupling, 2)
        * (grad_list_2[13] / (np.power(nt_hooft, (len(loop_list[13]) + 2) / 2)))
    )

    return np.concatenate((grad_1.real, grad_2.real), axis=0)


def print_results(
    omega_length,
    loop_list,
    non_zero,
    nonzero_index,
    nonzero_y,
    non_zero_lo,
    nonzero_loop1_index,
    nonzero_loop2_index,
    nonzero_z,
    adjoint_loops,
    max_size,
    omega_size,
    N,
    m,
    g_YM,
    loop,
    omega,
    little_omega,
    start,
    Energy,
    end,
    run_list,
    GTOL,
):

    matrix_1_final = np.diag(Energy.x[0:N])
    temp_matrix_2_final = np.reshape(Energy.x[N : N ** 2 + N], (N, N))
    real_matrix_2_final = np.triu(temp_matrix_2_final)
    real_matrix_2_final = (
        real_matrix_2_final
        + real_matrix_2_final.T
        - np.diag(np.diag(temp_matrix_2_final))
    )
    imag_matrix_2_final = np.tril(temp_matrix_2_final)
    imag_matrix_2_final = imag_matrix_2_final - imag_matrix_2_final.T
    matrix_2_final = real_matrix_2_final + imag_matrix_2_final * 1.0j

    matrices_array_final = np.zeros((2, N, N), dtype=complex)
    matrices_array_final[0] = matrix_1_final
    matrices_array_final[1] = matrix_2_final

    # Loops calculated with derivatives
    loop[0] = 1.0
    loop[1] = np.sum(Energy.x[0:N]) / (np.power(N, 1.5))
    loop[2] = np.trace(matrices_array_final[1]) / (np.power(N, 1.5))

    grad_array_1 = np.zeros((max_size + 1, N, N), dtype=complex)
    grad_array_2 = np.zeros((max_size + 1, N, N), dtype=complex)
    grad_list_1 = np.zeros((max_size + 1, N * N), dtype=complex)
    grad_list_2 = np.zeros((max_size + 1, N * N), dtype=complex)
    grad_array_1[1] = np.identity(N) / (np.power(N, 1.5))
    grad_array_2[2] = np.identity(N) / (np.power(N, 1.5))
    grad_list_1[1] = (grad_array_1[1]).flatten()
    grad_list_2[2] = (grad_array_2[2]).flatten()

    for i in range(3, max_size + 1):  # assumes first 3 loops will alwyas be the same
        for k in range(len(loop_list[i])):
            loop_left = loop_list[i][0:k]
            loop_right = loop_list[i][k + 1 : len(loop_list[i])]
            loop_grad = np.concatenate((loop_right, loop_left))
            grad_len = len(loop_grad)
            grad_matrix = matrices_array_final[loop_grad[0] - 1]

            if grad_len > 1:
                for l in range(1, grad_len):
                    if loop_grad[l] == 1:
                        grad_matrix = (
                            grad_matrix * Energy.x[0:N]
                        )  # [k]#matrices_list[loop_list[i][k]-1]
                    else:
                        grad_matrix = np.dot(grad_matrix, matrices_array_final[1])
            if loop_list[i][k] == 1:
                # grad_list_1[i]+=np.diagonal(grad_matrix.real)
                grad_array_1[i] += grad_matrix
            else:
                grad_array_2[i] += grad_matrix

        last_index = len(loop_list[i]) - 1
        if loop_list[i][last_index] == 1:
            loop_matrix = grad_matrix * Energy.x[0:N]
        else:
            loop_matrix = np.dot(grad_matrix, matrices_array_final[1])  # [

        loop[i] = np.trace(loop_matrix) / (np.power(N, (len(loop_list[i]) + 2) / 2))

        upper_matrix_1 = np.triu((grad_array_1[i] + grad_array_1[i].T)) / np.sqrt(
            2
        ) + np.diag(np.diag(grad_array_1[i])) * (1 - np.sqrt(2))
        lower_matrix_1 = (
            -1.0j * (np.tril((grad_array_1[i] - grad_array_1[i].T))) / np.sqrt(2)
        )
        grad_array_1[i] = (upper_matrix_1 + lower_matrix_1) / (
            np.power(N, (len(loop_list[i]) + 2) / 2)
        )
        grad_list_1[i] = grad_array_1[i].flatten()

        upper_matrix_2 = np.triu((grad_array_2[i] + grad_array_2[i].T)) / np.sqrt(
            2
        ) + np.diag(np.diag(grad_array_2[i])) * (1 - np.sqrt(2))
        lower_matrix_2 = (
            -1.0j * (np.tril((grad_array_2[i] - grad_array_2[i].T))) / np.sqrt(2)
        )
        grad_array_2[i] = (upper_matrix_2 + lower_matrix_2) / (
            np.power(N, (len(loop_list[i]) + 2) / 2)
        )
        grad_list_2[i] = grad_array_2[i].flatten()

    grad_master = np.zeros((max_size, 2 * N * N), dtype=complex)
    for i in range(1, max_size + 1):
        grad_master[i - 1] = np.concatenate((grad_list_1[i], grad_list_2[i]), axis=0)

    print()
    print("m =", m, ",", "g_YM=", g_YM)
    print(
        "No of loops =",
        max_size,
        ",",
        "lmax =",
        omega_length,
        ",",
        "Omega is",
        omega_size,
        "x",
        omega_size,
        "matrix",
    )
    print("N =", N, ",", "No of master variables = ", N * (N + 1))
    # np.format_float_scientific(np.amin(np.abs(np.concatenate((grad_1.real,grad_2.real),axis=0))),precision=6)

    print(
        "Minimization time",
        ",",
        datetime.timedelta(seconds=int(end - start)),
        "(days,h:m:s)",
    )
    print()
    smallest_grad_mod = np.format_float_scientific(
        np.amin(np.abs(Energy.jac)), precision=6
    )

    largest_grad_mod = np.format_float_scientific(
        np.amax(np.abs(Energy.jac)), precision=6
    )

    print(
        "Smallest gradient component modulus:",
        ",",
        smallest_grad_mod,
    )
    print(
        "Largest gradient component modulus:",
        ",",
        largest_grad_mod,
    )

    # gtol_final=np.sum (np.absolute(Energy.jac))
    print()
    # np.sqrt(np.sum(np.power(Energy.jac,2))/(N*(N+1)))
    print(
        "Final gtol :",
        ",",
        np.format_float_scientific(
            np.sqrt(np.sum(np.power(Energy.jac, 2))), precision=6
        ),
    )
    print()
    print(
        "Final normalized gtol :",
        ",",
        np.format_float_scientific(
            np.sqrt(np.sum(np.power(Energy.jac, 2)) / (np.sqrt(N * (N + 1)))),
            precision=6,
        ),
    )

    # print()
    # print("Final gtol attribute ? :",",",Energy.gtol)

    print()
    print("Energy:", ",", np.format_float_scientific(Energy.fun, precision=6))

    # print ("Energy =", Energy.fun)
    for i in range(max_size + 1):
        print(
            "Loop,",
            i,
            ",=,",
            loop_list[i],
            ",has value,",
            format(loop[i].real, ".6f"),
            ",imaginary part,",
            format(loop[i].imag, ".6f"),
        )  # round (loop[i].real,5)

    # omega=np.zeros ((omega_dim,omega_dim),dtype=complex)
    for i in range(omega_size):
        for j in range(omega_size):
            omega[i, j] = 0.0
            for indx in range(non_zero[i + 1, j + 1]):
                k = nonzero_index[i + 1, j + 1, indx]
                y = nonzero_y[i + 1, j + 1, indx]
                omega[i, j] += y * loop[k]

    print()
    # print ("Explicitly Symmetric Omega to accuracy 1e-15 ? " ,",", np.all ((np.isclose(omega.T, omega,atol=1e-15 ))))
    # print ("Explicitly Symmetric Omega to accuracy 1e-16 ? " ,",", np.all ((np.isclose(omega.T, omega,atol=1e-16 ))))
    symmetric_omega_accuracy = np.all((np.isclose(omega.T, omega, atol=1e-18)))
    print(
        "Symmetric Omega evaluated. Is is symmetric to accuracy 1e-18 ? ",
        symmetric_omega_accuracy,
    )
    print()

    print("Symmetric Omega eigenvalues - real part")
    Eigen = np.sort(linalg.eigvals(omega))
    i = 0
    for x in list(Eigen):
        print(
            "{:.8f}".format(Eigen[i].real),
            "    ",
            np.format_float_scientific(Eigen[i].real, precision=6),
        )
        i += 1

    print()
    largest_imag_eigenvalue = np.format_float_scientific(
        np.amax(np.abs(Eigen.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue,
    )

    grad_master_res = grad_master[0:omega_size, :]
    omega_first = np.dot(grad_master_res, grad_master_res.T)

    print()

    print(
        "Eigenvalues of Symmetric Omega constructed with restricted set of master variables  - real part"
    )
    Eigen1 = np.sort(linalg.eigvals(omega_first) * N * N)
    i = 0
    for x in list(Eigen1):
        print(
            "{:.8f}".format(Eigen1[i].real),
            "    ",
            np.format_float_scientific(Eigen1[i].real, precision=6),
        )
        i += 1

    print()
    largest_imag_eigenvalue_2 = np.format_float_scientific(
        np.amax(np.abs(Eigen1.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue_2,
    )

    print()
    # print ("Explicitly Symmetric Omega to accuracy 1e-15 ? " ,",", np.all ((np.isclose(omega.T, omega,atol=1e-15 ))))
    # print ("Explicitly Symmetric Omega to accuracy 1e-16 ? " ,",", np.all ((np.isclose(omega.T, omega,atol=1e-16 ))))
    omega_compare = np.all(
        (
            np.isclose(
                list(np.sort(linalg.eigvals(omega).real)),
                list(np.sort(linalg.eigvals(omega_first).real) * N * N),
                atol=1e-18,
            )
        )
    )
    print(
        "Are Omega and reconstructed Symmetric Omega real eigenvalues same to accuracy 1e-18 ? ",
        omega_compare,
    )
    # print()

    h_omega = np.zeros((omega_size, omega_size), dtype=complex)
    for i in range(omega_size):
        for j in range(omega_size):
            h_omega[i, j] = 0.0
            for indx in range(non_zero[i + 1, adjoint_loops[j + 1]]):
                k = nonzero_index[i + 1, adjoint_loops[j + 1], indx]
                y = nonzero_y[i + 1, adjoint_loops[j + 1], indx]
                h_omega[i, j] += y * loop[k]

    print()
    print("Hermitian Omega eigenvalues - real part")
    Eigen2 = np.sort(linalg.eigvals(h_omega))
    i = 0
    for x in list(Eigen2):
        print(
            "{:.8f}".format(Eigen2[i].real),
            "    ",
            np.format_float_scientific(Eigen2[i].real, precision=6),
        )
        i += 1

    print()
    largest_imag_eigenvalue_3 = np.format_float_scientific(
        np.amax(np.abs(Eigen2.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue_3,
    )

    # print ("h_omega hermitian ? ", np.array_equal (np.conj(h_omega), h_omega.T))
    print()
    confirm_hermitian = np.all((np.isclose(np.conj(h_omega), h_omega.T, atol=1e-18)))
    print(
        "Confirm hermiticity to accuracy 1e-18 ?  ",
        confirm_hermitian,
    )

    """ Big Omega """

    print()
    print(
        "Eigenvalues of Symmetric Big Omega (constructed with master variables)  - real part only"
    )

    big_omega = multi_dot([grad_master, grad_master.T]) * N * N

    Eigen3 = np.sort(linalg.eigvals(big_omega))
    i = 0
    for x in list(Eigen3):
        print(
            "{:.8f}".format(Eigen3[i].real),
            "    ",
            np.format_float_scientific(Eigen3[i].real, precision=6),
        )
        i += 1

    print()
    largest_imag_eigenvalue_4 = np.format_float_scientific(
        np.amax(np.abs(Eigen3.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue_4,
    )

    # print ("Eigenvalues of Symmetric Big Omega constructed with master variables  - real part only - decimal ")
    # for x in list(np.sort(linalg.eigvals(multi_dot([grad_master,grad_master.T])).real)*N*N):
    # print (np.format_float_scientific(x,precision=6))
    # print ("{:.8f}". format(x))

    print()
    print(
        "Eigenvalues of Hermitian Big Omega (constructed with master variables)  - real part only "
    )

    h_big_omega = np.zeros((max_size, max_size), dtype=complex)
    for i in range(max_size):
        for j in range(max_size):
            h_big_omega[i, j] = big_omega[i, adjoint_loops[j + 1] - 1]

    Eigen4 = np.sort(linalg.eigvals(h_big_omega))
    i = 0
    for x in list(Eigen4):
        print(
            "{:.8f}".format(Eigen4[i].real),
            "    ",
            np.format_float_scientific(Eigen4[i].real, precision=6),
        )
        i += 1

    print()

    largest_imag_eigenvalue_5 = np.format_float_scientific(
        np.amax(np.abs(Eigen4.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue_5,
    )

    """ Start spectrum calculations  """

    A = np.zeros((omega_size, omega_size), dtype=complex)
    for i in range(omega_size):
        for indx in range(non_zero_lo[i + 1]):
            j = nonzero_loop1_index[i + 1, indx]
            k = nonzero_loop2_index[i + 1, indx]
            z = nonzero_z[i + 1, indx]
            if j != 0:
                A[i, j - 1] += z * loop[k]  # axis?
            if k != 0:
                A[i, k - 1] += z * loop[j]

    A_2 = np.zeros((omega_size, omega_size, omega_size), dtype=int)
    for i in range(omega_size):
        for indx in range(non_zero_lo[i + 1]):
            j = nonzero_loop1_index[i + 1, indx]
            k = nonzero_loop2_index[i + 1, indx]
            z = nonzero_z[i + 1, indx]
            if j != 0 and k != 0:
                A_2[i, j - 1, k - 1] += z  # axis?
            if k != 0 and j != 0:
                A_2[i, k - 1, j - 1] += z

    omega_grad = np.zeros((omega_size, omega_size, omega_size), dtype=int)
    for i in range(omega_size):
        for j in range(omega_size):
            for indx in range(non_zero[i + 1, j + 1]):
                k = nonzero_index[i + 1, j + 1, indx]
                y = nonzero_y[i + 1, j + 1, indx]
                if k != 0 and k <= omega_size:
                    omega_grad[i, j, k - 1] += y

    # little_omega=np.zeros ((omega_size),dtype=complex)
    for i in range(omega_size):
        little_omega[i] = 0.0
        for indx in range(non_zero_lo[i + 1]):
            j = nonzero_loop1_index[i + 1, indx]
            k = nonzero_loop2_index[i + 1, indx]
            z = nonzero_z[i + 1, indx]
            little_omega[i] += z * loop[j] * loop[k]

    LnJ = np.linalg.solve(omega, little_omega)

    omega_inv = linalg.inv(omega)

    print()
    print(" SPECTRUM calculations")
    print()
    print("Loop based fluctuations spectrum")

    V_2 = multi_dot([A.T, omega_inv, A])

    V_2 += np.sum(
        np.array(
            [
                [
                    [A_2[i, k, l] * LnJ[i] for k in range(omega_size)]
                    for l in range(omega_size)
                ]
                for i in range(omega_size)
            ]
        ),
        axis=0,
    )

    omega_deriv = np.sum(
        np.array(
            [
                [
                    [omega_grad[i, j, k] * LnJ[j] for k in range(omega_size)]
                    for i in range(omega_size)
                ]
                for j in range(omega_size)
            ]
        ),
        axis=0,
    )

    V_2 += multi_dot([omega_deriv.T, omega_inv, omega_deriv])

    V_2 -= multi_dot([A.T, omega_inv, omega_deriv]) + multi_dot(
        [omega_deriv.T, omega_inv, A]
    )

    v_2 = multi_dot([omega, V_2]) / 4.0

    Eigen5 = np.sort(np.sqrt(np.absolute(linalg.eigvals(v_2).real)))
    i = 0
    for x in list(Eigen5):
        print(
            "{:.8f}".format(Eigen5[i].real),
            "    ",
            np.format_float_scientific(Eigen5[i].real, precision=6),
        )
        i += 1

    print()
    print("Transpose of loop mass matrix")
    v_2 = multi_dot([V_2, omega]) / 4.0
    Eigen6 = np.sort(np.sqrt(np.absolute(linalg.eigvals(v_2).real)))
    i = 0
    for x in list(Eigen6):
        print(
            "{:.8f}".format(Eigen6[i].real),
            "    ",
            np.format_float_scientific(Eigen6[i].real, precision=6),
        )
        i += 1

    """     ---------------------------------------------------------------------------
    Master variabels spectrum
    """

    print()
    print(
        "Spectrum with restricted set of Master variables - should agree with loop based spectrum"
    )

    print()
    print("Eigenvalues of squared mass matrix")
    # sub_grad_list=grad_list[1:omega_size+1,:]
    # print (sub_grad_list.shape)

    # master_square = np.zeros((2*N*N,2*N*N), dtype=complex)

    i = 0
    Eigen7x = (
        np.sort(
            linalg.eigvals(multi_dot([grad_master_res.T, V_2, grad_master_res])).real
        )
        * N
        * N
        / 4.0
    )
    for x in list(Eigen7x):
        if i == 0:
            print(
                "{:.8f}".format(x),
                "   ",
                np.format_float_scientific(x, precision=6),
                "\n   .....  ",
            )
        if i > 2 * N * N - omega_size - 2:
            print("{:.8f}".format(x), "   ", np.format_float_scientific(x, precision=6))
        i += 1
    # print (np.format_float_scientific(x,precision=6),",")

    Eigen7 = (
        linalg.eigvals(multi_dot([grad_master_res.T, V_2, grad_master_res]))
        * N
        * N
        / 4.0
    )
    print()
    largest_imag_eigenvalue_6 = np.format_float_scientific(
        np.amax(np.abs(Eigen7.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue_6,
    )

    print()
    print(
        "Spectrum wrt restricted set of master variables - taken absolute value of mass squared eigenvalues"
    )
    i = 0
    Eigen8x = (
        np.sort(
            np.sqrt(
                np.absolute(
                    linalg.eigvals(
                        multi_dot([grad_master_res.T, V_2, grad_master_res])
                    ).real
                )
            )
        )
        * N
        / 2.0
    )

    for x in list(Eigen8x):
        if i == 0:
            print(
                "{:.8f}".format(x),
                "   ",
                np.format_float_scientific(x, precision=6),
                "\n   .....  ",
            )
        if i > 2 * N * N - omega_size - 2:
            print("{:.8f}".format(x), "   ", np.format_float_scientific(x, precision=6))
        i += 1
    # print (np.format_float_scientific(x,precision=6),",")
    print()
    print("Master variables spectrum with FULL set of loops")

    V_2_all = np.zeros((max_size, max_size), dtype=complex)
    V_2_all[0:omega_size, 0:omega_size] = multi_dot([A.T, omega_inv, A])
    V_2_all[0:omega_size, 0:omega_size] += np.sum(
        np.array(
            [
                [
                    [A_2[i, k, l] * LnJ[i] for k in range(omega_size)]
                    for l in range(omega_size)
                ]
                for i in range(omega_size)
            ]
        ),
        axis=0,
    )

    omega_grad_all = np.zeros((omega_size, omega_size, max_size), dtype=int)
    for i in range(omega_size):
        for j in range(omega_size):
            for indx in range(non_zero[i + 1, j + 1]):
                k = nonzero_index[i + 1, j + 1, indx]
                y = nonzero_y[i + 1, j + 1, indx]
                if k != 0:
                    omega_grad_all[i, j, k - 1] += y

    omega_deriv = np.sum(
        np.array(
            [
                [
                    [omega_grad_all[i, j, k] * LnJ[j] for k in range(max_size)]
                    for i in range(omega_size)
                ]
                for j in range(omega_size)
            ]
        ),
        axis=0,
    )

    V_2_all += multi_dot([omega_deriv.T, omega_inv, omega_deriv])

    V_2_all[0:omega_size, 0:max_size] -= multi_dot([A.T, omega_inv, omega_deriv])
    V_2_all[0:max_size, 0:omega_size] -= multi_dot([omega_deriv.T, omega_inv, A])

    # v_2=multi_dot ([omega,V_2])/4.0
    print()
    print("Eigenvalues of squared mass matrix")

    i = 0
    Eigen9x = (
        np.sort(linalg.eigvals(multi_dot([grad_master.T, V_2_all, grad_master])).real)
        * N
        * N
        / 4.0
    )
    for x in list(Eigen9x):
        if i == 0:
            print(
                "{:.8f}".format(x),
                "   ",
                np.format_float_scientific(x, precision=6),
                "\n   .....  ",
            )
        if i > 2 * N * N - omega_size - 2:
            print("{:.8f}".format(x), "   ", np.format_float_scientific(x, precision=6))
        i += 1
    # print (np.format_float_scientific(x,precision=6),",")

    Eigen8 = (
        linalg.eigvals(multi_dot([grad_master.T, V_2_all, grad_master])) * N * N / 4.0
    )
    print()
    largest_imag_eigenvalue_1 = np.format_float_scientific(
        np.amax(np.abs(Eigen8.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue_1,
    )

    print()
    print(
        "Spectrum wrt ALL master variables - taken absolute value of mass squared eigenvalues"
    )
    i = 0
    Eigen10x = (
        np.sort(
            np.sqrt(
                np.absolute(
                    linalg.eigvals(
                        multi_dot([grad_master.T, V_2_all, grad_master])
                    ).real
                )
            )
        )
        * N
        / 2.0
    )
    for x in list(Eigen10x):
        if i == 0:
            print(
                "{:.8f}".format(x),
                "   ",
                np.format_float_scientific(x, precision=6),
                "\n   .....  ",
            )
        if i > 2 * N * N - omega_size - 2:
            print("{:.8f}".format(x), "   ", np.format_float_scientific(x, precision=6))
        i += 1

    print()
    print("Square of Big Omega loop based spectrum - Big Omega on the right")
    print()
    # print ("(taken absolute value of mass squared matrix eigenvalues)")
    i = 0
    Eigen11x = (
        np.sort(linalg.eigvals(multi_dot([V_2_all, grad_master, grad_master.T])).real)
        * N
        * N
        / 4.0
    )
    for x in list(Eigen11x):
        if i == 0:
            print(
                "{:.8f}".format(x),
                "   ",
                np.format_float_scientific(x, precision=6),
                "\n   .....  ",
            )
        if i > max_size - omega_size - 2:
            print("{:.8f}".format(x), "   ", np.format_float_scientific(x, precision=6))
        i += 1

    Eigen9 = (
        linalg.eigvals(multi_dot([V_2_all, grad_master, grad_master.T])) * N * N / 4.0
    )
    print()
    largest_imag_eigenvalue_7 = np.format_float_scientific(
        np.amax(np.abs(Eigen9.imag)), precision=6
    )
    print(
        "Largest imaginary part of eigenvalues (mod) :    ",
        largest_imag_eigenvalue_7,
    )

    print()
    print("Big Omega loop based spectrum - Big Omega on the right")
    print()
    # print ("(taken absolute value of mass squared matrix eigenvalues)")
    i = 0
    Eigen12x = (
        np.sort(
            np.sqrt(
                np.absolute(
                    linalg.eigvals(
                        multi_dot([V_2_all, grad_master, grad_master.T])
                    ).real
                )
            )
        )
        * N
        / 2.0
    )
    for x in list(Eigen12x):
        if i == 0:
            print(
                "{:.8f}".format(x),
                "   ",
                np.format_float_scientific(x, precision=6),
                "\n   .....  ",
            )
        if i > max_size - omega_size - 2:
            print("{:.8f}".format(x), "   ", np.format_float_scientific(x, precision=6))
        i += 1

    keys = run_list[0].keys()
    results_csv_name = "lmax_{}_N_{}_m_{}_g_YM_{}_gtol_{:,.1E}.csv".format(
        omega_length, N, m, g_YM, GTOL / np.sqrt(N * (N + 1))
    )
    PATH = pathlib.Path(__file__).parent.absolute()
    current_folder = str(PATH)
    results_folder_name = "results"
    results_folder = os.path.join(current_folder, results_folder_name)
    results_csv = os.path.join(results_folder, results_csv_name)
    os.makedirs(results_folder, exist_ok=True)

    print("Results saved on csv file in results folder with name: \n", results_csv_name)
    with open(results_csv, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(run_list)

        writer = csv.writer(output_file)
        writer.writerow([])
        writer.writerow(["m = {}".format(m), "g_yang_mills = {}".format(g_YM)])
        writer.writerow(
            ["No of loops = {}".format(max_size), "lmax = {}".format(omega_length)]
        )
        writer.writerow(
            [
                "Omega is a {} x {}".format(omega_size, omega_size),
                "gtol: [sqrt(N*(N+1))]*{:.1E}".format(GTOL / (np.sqrt(N * (N + 1)))),
            ]
        )
        writer.writerow(
            ["N = {}".format(N), "#Master variables = {}".format(N * (N + 1))]
        )
        writer.writerow(
            [
                "Energy = ",
                "{}\n".format(np.format_float_scientific(Energy.fun, precision=6)),
            ]
        )
        writer.writerow([])
        writer.writerow(
            ["Smallest gradient component modulus = ", "{}".format(smallest_grad_mod)]
        )
        writer.writerow(
            ["Largest gradient component modulus = ", "{}".format(largest_grad_mod)]
        )
        writer.writerow([])
        for i in range(max_size + 1):
            writer.writerow(
                [
                    "Loop[{}] = {} ".format(i, loop_list[i]),
                    "has value",
                    "{}".format(format(loop[i].real, ".6f")),
                    "imaginary part",
                    "{}".format(format(loop[i].imag, ".6f")),
                ]
            )
        writer.writerow([])
        writer.writerow(
            [
                "Symmetric Omega evaluated. Is is symmetric to accuracy 1e-18 ?",
                symmetric_omega_accuracy,
            ]
        )
        writer.writerow([])

        writer.writerow(["Symmetric Omega eigenvalues - real part"])
        i = 0
        for x in list(Eigen):
            writer.writerow(
                [
                    np.format_float_scientific(Eigen[i].real, precision=6),
                ]
            )
            i += 1

        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) :    ",
                largest_imag_eigenvalue,
            ]
        )

        writer.writerow([])
        writer.writerow(
            [
                "Eigenvalues of Symmetric Omega constructed with restricted set of master variables",
                "real part",
            ]
        )
        i = 0
        for x in list(Eigen1):
            writer.writerow([np.format_float_scientific(Eigen2[i].real, precision=6)])
            i += 1

        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) : ",
                largest_imag_eigenvalue_2,
            ]
        )

        writer.writerow([])
        writer.writerow(
            [
                "Are Omega and reconstructed Symmetric Omega real eigenvalues same to accuracy 1e-18 ? ",
                omega_compare,
            ]
        )

        writer.writerow([])
        writer.writerow(["Hermitian Omega eigenvalues - real part"])
        i = 0
        for x in list(Eigen2):
            writer.writerow([np.format_float_scientific(Eigen1[i].real, precision=6)])
            i += 1

        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) : ",
                largest_imag_eigenvalue_3,
            ]
        )

        writer.writerow([])
        writer.writerow(
            ["Confirm hermiticity to accuracy 1e-18 ?  ", confirm_hermitian]
        )

        writer.writerow([])
        writer.writerow(
            [
                "Eigenvalues of Symmetric Big Omega (constructed with master variables)",
                "real part only",
            ]
        )
        i = 0
        for x in list(Eigen3):
            writer.writerow([np.format_float_scientific(Eigen3[i].real, precision=6)])
            i += 1
        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) :    ",
                largest_imag_eigenvalue_4,
            ]
        )
        writer.writerow([])
        writer.writerow(
            [
                "Eigenvalues of Hermitian Big Omega (constructed with master variables)",
                "- real part only ",
            ]
        )
        i = 0
        for x in list(Eigen4):
            writer.writerow([np.format_float_scientific(Eigen4[i].real, precision=6)])
            i += 1

        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) :    ",
                largest_imag_eigenvalue_5,
            ]
        )
        writer.writerow([])
        writer.writerow(["SPECTRUM calculations"])
        writer.writerow([])
        writer.writerow(["Loop based fluctuations spectrum"])
        i = 0
        for x in list(Eigen5):
            writer.writerow([np.format_float_scientific(Eigen5[i].real, precision=6)])
            i += 1
        writer.writerow([])
        writer.writerow(["Transpose of loop mass matrix"])
        i = 0
        for x in list(Eigen6):
            writer.writerow([np.format_float_scientific(Eigen6[i].real, precision=6)])
            i += 1
        writer.writerow([])
        writer.writerow(
            [
                "Spectrum with restricted set of Master variables "
                + "- should agree with loop based spectrum"
            ]
        )
        writer.writerow([])
        writer.writerow(["Eigenvalues of squared mass matrix"])
        i = 0
        for x in list(Eigen7x):
            if i == 0:
                writer.writerow([np.format_float_scientific(x, precision=6)])
                writer.writerow([" ....... "])
            elif i > 2 * N * N - omega_size - 2:
                writer.writerow([np.format_float_scientific(x, precision=6)])
            i += 1

        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) :    ",
                largest_imag_eigenvalue_6,
            ]
        )
        writer.writerow([])
        writer.writerow(
            [
                "Spectrum wrt restricted set of master variables ",
                "taken absolute value of mass squared eigenvalues",
            ]
        )
        i = 0
        for x in list(Eigen8x):
            if i == 0:
                writer.writerow([np.format_float_scientific(x, precision=6)])
                writer.writerow([" ....... "])
            elif i > 2 * N * N - omega_size - 2:
                writer.writerow([np.format_float_scientific(x, precision=6)])
            i += 1
        writer.writerow([])
        writer.writerow(["Master variables spectrum with FULL set of loops"])
        writer.writerow([])
        writer.writerow(["Eigenvalues of squared mass matrix"])
        i = 0
        for x in list(Eigen9x):
            if i == 0:
                writer.writerow([np.format_float_scientific(x, precision=6)])
                writer.writerow([" ....... "])
            elif i > 2 * N * N - omega_size - 2:
                writer.writerow([np.format_float_scientific(x, precision=6)])
            i += 1
        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) :    ",
                largest_imag_eigenvalue_1,
            ]
        )
        writer.writerow([])
        writer.writerow(
            [
                "Spectrum wrt ALL master variables ",
                " taken absolute value of mass squared eigenvalues",
            ]
        )
        i = 0
        for x in list(Eigen10x):
            if i == 0:
                writer.writerow([np.format_float_scientific(x, precision=6)])
                writer.writerow([" ....... "])
            elif i > 2 * N * N - omega_size - 2:
                writer.writerow([np.format_float_scientific(x, precision=6)])
            i += 1
        writer.writerow([])
        writer.writerow(
            ["Square of Big Omega loop based spectrum - Big Omega on the right"]
        )
        writer.writerow([])
        i = 0
        for x in list(Eigen11x):
            if i == 0:
                writer.writerow([np.format_float_scientific(x, precision=6)])
                writer.writerow([" ....... "])
            elif i > max_size - omega_size - 2:
                writer.writerow([np.format_float_scientific(x, precision=6)])
            i += 1
        writer.writerow([])
        writer.writerow(
            [
                "Largest imaginary part of eigenvalues (mod) :    ",
                largest_imag_eigenvalue_7,
            ]
        )
        writer.writerow([])
        writer.writerow(["Big Omega loop based spectrum - Big Omega on the right"])
        writer.writerow([])
        i = 0
        for x in list(Eigen12x):
            if i == 0:
                writer.writerow([np.format_float_scientific(x, precision=6)])
                writer.writerow([" ....... "])
            elif i > max_size - omega_size - 2:
                writer.writerow([np.format_float_scientific(x, precision=6)])
            i += 1
