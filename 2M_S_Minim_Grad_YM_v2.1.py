# attempt to get spectrum from master variables
#   ?? this code loads _collective_info_v3 files
# changed normalization of Gradient

import pandas as pd
import numpy as np
from scipy import linalg
from itertools import combinations
from scipy.optimize import minimize
from numpy.linalg import multi_dot
from timeit import default_timer as timer
import datetime
import sys
import os
import pathlib
import csv

from Collectivev2 import initialize_loops
from Collectivev2 import compare, cyclic_generator
from Collectivev2 import effect_pot, effect_pot_grad


"""
  -------------       Main program    --------------
"""

lmax = sys.argv[1] if len(sys.argv) > 1 else input("Enter lmax :    ")
omega_length = int(lmax)
file_name = "lmax_" + str(omega_length) + "_collective_info_v2.npy"
print(
    "\n Importing list of loops, Omega and little omega coefficients from file :",
    file_name,
)

PATH = pathlib.Path(
    __file__
).parent.absolute()  # get the parent directory of the script
current_folder = str(PATH)
data_folder_name = "data"
data_folder = os.path.join(current_folder, data_folder_name)
data_file_name = "lmax_" + str(omega_length) + "_collective_info_v2.npy"
data_file = os.path.join(data_folder, data_file_name)

with open(data_file, "rb") as f:
    omega_length = np.load(f)
    max_length = np.load(f)
    numb_tuples_list = np.load(f, allow_pickle=True)
    loop_list = np.load(f, allow_pickle=True)
    non_zero = np.load(f, allow_pickle=True)
    nonzero_index = np.load(f, allow_pickle=True)
    nonzero_y = np.load(f, allow_pickle=True)
    non_zero_lo = np.load(f, allow_pickle=True)
    nonzero_loop1_index = np.load(f, allow_pickle=True)
    nonzero_loop2_index = np.load(f, allow_pickle=True)
    nonzero_z = np.load(f, allow_pickle=True)
    adjoint_loops = np.load(f, allow_pickle=True)

max_size = numb_tuples_list[max_length][1]
omega_size = numb_tuples_list[omega_length][1]
little_omega_size = numb_tuples_list[omega_length - 2][1]

print(
    "File import completed. Total number of loops is {}. Omega is a {} x {} matrix".format(
        max_size, omega_size, omega_size
    )
)
NString = sys.argv[2] if len(sys.argv) > 2 else input("Enter 't Hooft's N :    ")
N = int(NString)
print("Number of master variables :", N * (N + 1), "\n")
mstring = (
    sys.argv[3]
    if len(sys.argv) > 3
    else input("Enter a value for the mass parameter m :   ")
)
m = float(mstring)
g_YM_string = (
    sys.argv[4]
    if len(sys.argv) > 4
    else input("Enter a value for the Yang Mills coupling g_YM :   ")
)
g_YM = float(g_YM_string)

gtol_string = input("Enter starting GTOL :[√(N*(N+1))]*1.0E-")
gtol_float = float(gtol_string)
gtol = lambda x: np.sqrt(N * (N + 1)) * 10.0 ** ((-x))
print("{:,.1E}".format(gtol(gtol_float) / (np.sqrt(N * (N + 1)))))
GTOL = gtol(gtol_float)


inputs_folder_name = "inputs"
inputs_folder = os.path.join(current_folder, inputs_folder_name)
os.makedirs(inputs_folder, exist_ok=True)
inputs_file_name = "lmax_{}_N_{}_m_{}_g_YM_{}_gtol_{:,.1E}.npy".format(
    lmax, N, m, g_YM, GTOL / np.sqrt(N * (N + 1))
)
inputs_file = os.path.join(inputs_folder, inputs_file_name)


load = input("Start minimization from saved file? Y/n: \n").lower().strip()
if load == "y":
    try:
        x_init = np.load(inputs_file)
        print("Input file loaded successfully \n")
        print("Minimization will start with:")
    except IOError:
        print(
            "Input file not found, minimization will start from a random distribution with:"
        )
        eigen_init = (np.random.rand(N) - np.ones(N) * 0.5) * np.sqrt(N)
        m_2_init = (np.random.rand(N, N) - np.ones((N, N)) * 0.5) * np.sqrt(N)
        x_init = np.concatenate((eigen_init, m_2_init.flatten()), axis=0)
else:
    eigen_init = (np.random.rand(N) - np.ones(N) * 0.5) * np.sqrt(N)
    m_2_init = (np.random.rand(N, N) - np.ones((N, N)) * 0.5) * np.sqrt(N)
    x_init = np.concatenate((eigen_init, m_2_init.flatten()), axis=0)
    print("Starting minimization from random distribution with:")

loop, omega, little_omega, LnJ = initialize_loops(omega_size, max_size)


ARGS = (
    omega_size,
    max_size,
    little_omega_size,
    N,
    m,
    g_YM,
    loop_list,
    non_zero,
    non_zero_lo,
    nonzero_index,
    nonzero_y,
    nonzero_z,
    nonzero_loop1_index,
    nonzero_loop2_index,
)


print("lmax = {}".format(lmax))
print("g_YM = {}".format(g_YM))
print("mass = {}, N = {}".format(m, N))
print("gtol = [√(N*(N+1))]*{:.1E}".format(GTOL / (np.sqrt(N * (N + 1)))))
input("Press Enter to start...")

start = timer()
Energy = minimize(
    effect_pot,
    x_init,
    args=ARGS,
    method="BFGS",
    jac=effect_pot_grad,
    options={"disp": True, "maxiter": 50000, "gtol": GTOL},
)
end = timer()
print("\t {:.3f} sec in minimization".format((end - start)))
print("\t Current gtol: [√(N*(N+1))]*{:.1E}".format(GTOL / (np.sqrt(N * (N + 1)))))
print("\t N = {}, No of master variables = {}".format(N, N * (N + 1)))
print("\t mass = {}, g_yang_mills = {} \n".format(m, g_YM))

# log run
run_list = []
run = {
    "Large N Energy": Energy.fun,
    "Iterations": Energy.nit,
    "Converged": Energy.success,
    " gtol [√(N*(N+1))]": GTOL / (np.sqrt(N * (N + 1))),
    "   N": N,
    "lmax": omega_length,
    "mass": m,
    "g_YM": g_YM,
}
run_list.append(run)


# save results of 1st minimization
Converge = Energy.success
save_results = input("Save solution of the optimization? Y/n: ").lower().strip()
if save_results == "y" and Converge:  # save file only if convergence is achieved
    inputs_file_name = "lmax_{}_N_{}_m_{}_g_YM_{}_gtol_{:,.1E}.npy".format(
        lmax, N, m, g_YM, GTOL / np.sqrt(N * (N + 1))
    )
    inputs_file = os.path.join(inputs_folder, inputs_file_name)
    x_min = Energy.x
    # hess_inv = Energy.hess_inv
    with open(inputs_file, "wb") as fx:
        np.save(fx, x_min)
        # np.save(fx, hess_inv)
    print("Solution saved in inputs folder as : \n")
    print(inputs_file_name, "\n")


while Converge:
    # decrease gtol by factor of 10 and repeat
    reply = (
        input(
            "Decrease gtol to [√(N*(N+1))]*{:.1E} and restart minimization from last solution? Y/n: ".format(
                GTOL / (10 * np.sqrt(N * (N + 1)))
            )
        )
        .lower()
        .strip()
    )
    if reply == "y":
        GTOL /= 10
        start = timer()
        x_min = Energy.x
        Energy = minimize(
            effect_pot,
            x_min,
            args=ARGS,
            method="CG",
            jac=effect_pot_grad,
            options={"disp": True, "maxiter": 50000, "gtol": GTOL},
        )
        end = timer()
        print("\t {:.3f} sec in minimization".format((end - start)))
        print(
            "\t Current gtol: [√(N*(N+1))]*{:.1E}".format(GTOL / (np.sqrt(N * (N + 1))))
        )
        print("\t N = {}, No of master variables = {}".format(N, N * (N + 1)))
        print("\t mass = {}, g_yang_mills = {} ".format(m, g_YM))
        Converge = Energy.success
        # log run
        run = {
            "Large N Energy": Energy.fun,
            "Iterations": Energy.nit,
            "Converged": Energy.success,
            " gtol [√(N*(N+1))]": GTOL / (np.sqrt(N * (N + 1))),
            "   N": N,
            "lmax": omega_length,
            "mass": m,
            "g_YM": g_YM,
        }
        run_list.append(run)
        print()
        save_results = input("Save solution of the optimization? Y/n: ").lower().strip()
        if (
            save_results == "y" and Converge
        ):  # save file only if convergence is achieved
            inputs_file_name = "lmax_{}_N_{}_m_{}_g_YM_{}_gtol_{:,.1E}.npy".format(
                lmax, N, m, g_YM, GTOL / np.sqrt(N * (N + 1))
            )
            inputs_file = os.path.join(inputs_folder, inputs_file_name)
            x_min = Energy.x
            # hess_inv = Energy.hess_inv
            with open(inputs_file, "wb") as fx:
                np.save(fx, x_min)
                # np.save(fx, hess_inv)
            print("Solution saved in inputs folder as : \n")
            print(inputs_file_name, "\n")

    elif reply == "n":

        input("Press Enter to show minimization history...")
        print()
        print("-" * 41)
        print("*" * 13, " Run history ", "*" * 13)
        print("-" * 41, "\n")
        df = pd.DataFrame(run_list)
        df[" gtol [√(N*(N+1))]"] = df[" gtol [√(N*(N+1))]"].map("{:,.1E}".format)
        df["Large N Energy"] = df["Large N Energy"].map("{:,.12f}".format)
        print(df)
        print()
        break
    else:
        print("Please enter Y/n.")

else:
    x_min = Energy.x
    if save_results == "y":
        inputs_file_name = "lmax_{}_N_{}_m_{}_g_YM_{}_gtol_{:,.1E}.npy".format(
            lmax, N, m, g_YM, GTOL / np.sqrt(N * (N + 1))
        )
        inputs_file = os.path.join(inputs_folder, inputs_file_name)
        os.makedirs(inputs_folder, exist_ok=True)
        with open(inputs_file, "wb") as fx:
            np.save(fx, x_min)
        print("x_min saved as {}\n".format(inputs_file_name))
    input("Press Enter to show minimization history...\n")
    print("-" * 41)
    print("*" * 13, " Run history ", "*" * 13)
    print("-" * 41, "\n")
    df = pd.DataFrame(run_list)
    df[" gtol [√(N*(N+1))]"] = df[" gtol [√(N*(N+1))]"].map("{:,.1E}".format)
    df["Large N Energy"] = df["Large N Energy"].map("{:,.12f}".format)
    print(df)
    print()


from Collectivev2 import print_results


results_input = input("Show Results and write to csv, Y/n:").lower()
if results_input == "y":
    print_results(
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
    )
else:
    print("Program has terminated")
