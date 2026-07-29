import json
import collections
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import math


def load_file(filepath: str):
    with open(filepath, "r") as file:
        data = json.load(file)
    # print(data)
    return data




data = load_file('../data/isoflops_curves.json')

# print(type(data))

# map from compute_budget to loss
optimal_mp = collections.defaultdict()

for curve in data:
    parameters = curve['parameters']
    compute_budget = curve['compute_budget']
    final_loss = curve['final_loss']
    if compute_budget not in optimal_mp:
        optimal_mp[compute_budget] = (parameters, final_loss)
    else:
        if final_loss < optimal_mp[compute_budget][1]:
            optimal_mp[compute_budget] = (parameters, final_loss)

# print(optimal_mp)

optimal_mp_with_datasetsz = collections.defaultdict()

for curve in optimal_mp.items():
    print(curve)
    compute, (params, loss) = curve
    dataset_sz = compute / (params * 6)
    print(dataset_sz)
    # optimal_mp[compute] = (params, dataset_sz, loss)
    optimal_mp_with_datasetsz[compute] = (params, dataset_sz, loss)

# compute -> params, dataset_sz, loss
print(optimal_mp_with_datasetsz)


# scipy.optimize.curve_fit uses non-linear least squares to fit a user-defined function to a set of data points
# curve_fit requires three inputs: model function, the independent variable array (x), and the dependent variable array (y)
compute_data = np.array([compute for compute in optimal_mp_with_datasetsz.keys()])
params_data = np.array([params for (params, _, _) in optimal_mp_with_datasetsz.values()])
datasetsz_data = np.array([dataset_sz for (_, dataset_sz, _) in optimal_mp_with_datasetsz.values()])
loss_data = np.array([loss for (_, _, loss) in optimal_mp_with_datasetsz.values()])


def model_func(compute, a, b):
    # y = a * x^b
    # x = compute, y = {params or dataset size or loss}
    return a * (compute ** b)


compute_loss_fits, x = curve_fit(model_func, compute_data, loss_data)
compute_params_fits, y = curve_fit(model_func, compute_data, params_data)
compute_datasetsz_fits, z = curve_fit(model_func, compute_data, datasetsz_data)

print(compute_params_fits) # [25.79297496  0.40381138]
print(x) # [[ 1.72274243e+03 -1.37578398e+00] [-1.37578398e+00  1.09936102e-03]]

# Calculations for problem a:
# y = 25.79297496 * (10^23)^0.40381138 = 5e+10
# y = 25.79297496 * (10^24)^0.40381138 = 1.27e+11

print(compute_datasetsz_fits) # [0.00633759 0.59677071]
print(z) # [[ 0.00011801 -0.00038051] [-0.00038051  0.00122729]]

# Calculations for problem b:
# y = 0.00633759 * (10^23)^0.59677071 = 3.37e+11
# y = 0.00633759 * (10^24)^0.59677071 = 1.33e+12

extrapolates = np.array([math.pow(10, 22), math.pow(10, 23), math.pow(10, 24)])
compute = np.append(compute_data, extrapolates)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# x = compute, y = loss
ax1.set_title("Loss vs. Compute")
ax1.plot(
    compute_data,
    loss_data,
    linestyle="",
    color="blue",
    marker="o",
    label="data",
)
ax1.plot(
    compute,
    model_func(compute, *compute_loss_fits),
    "r--",
    label=f"fit, a={compute_loss_fits[0]:.2f} (scale), b={compute_loss_fits[1]:.2f} (exponent)",
)
ax1.legend()
ax1.set_xlabel("Compute (FLOPs)")
ax1.set_xscale("log", base=10)
ax1.set_ylabel("Loss")
ax1.set_yscale("log", base=10)

ax2.set_title("Model size vs. Compute")
ax2.plot(
    compute_data,
    params_data,
    linestyle="",
    color="blue",
    marker="o",
    label="data",
)
ax2.plot(
    compute,
    model_func(compute, *compute_params_fits),
    "r--",
    label=f"fit, a={compute_params_fits[0]:.2f} (scale), b={compute_params_fits[1]:.2f} (exponent)",
)
ax2.legend()
ax2.set_xlabel("Compute (FLOPs)")
ax2.set_xscale("log", base=10)
ax2.set_ylabel("Num params")
ax2.set_yscale("log", base=10)

ax3.set_title("Dataset size vs. Compute")
ax3.plot(
    compute_data,
    datasetsz_data,
    linestyle="",
    color="blue",
    marker="o",
    label="data",
)
ax3.plot(
    compute,
    model_func(compute, *compute_datasetsz_fits),
    "r--",
    label=f"fit, a={compute_datasetsz_fits[0]:.2f} (scale), b={compute_datasetsz_fits[1]:.2f} (exponent)",
)
ax3.legend()
ax3.set_xlabel("Compute (FLOPs)")
ax3.set_xscale("log", base=10)
ax3.set_ylabel("Dataset size")
ax3.set_yscale("log", base=10)




plt.savefig('plot.png')

plt.show()
