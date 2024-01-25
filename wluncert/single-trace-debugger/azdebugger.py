import arviz as az
import numpy as np
from matplotlib import pyplot as plt

# Path to your NetCDF file
file_path = "arvizdata-partial-pooling.netcdf"

try:
    # Load the NetCDF file
    data = az.from_netcdf(file_path)

    # Data Overview
    print("Data Overview:")
    print(data)

    # Data Validation
    for group in data.groups():
        group_data = getattr(data, group)
        if np.isnan(group_data).any():
            print(f"Warning: NaN values found in the {group} group.")
        if np.isinf(group_data).any():
            print(f"Warning: Infinite values found in the {group} group.")

    # Summary Statistics
    summary = az.summary(data)
    print("Summary Statistics:")
    print(summary)

    # Diagnostic Checks
    print("Diagnostics:")
    diagnostics = az.rhat(data)
    print("\n\nR-hat statistics:")
    print(diagnostics)
    print("\n\nEffective Sample Size:")
    ess = az.ess(data)
    print(ess)

    # Visualization (Example: Histogram for a specific variable)
    filtered_variables = [var for var in data.posterior.data_vars if "hyper" in var]
    # filtered_variables = None
    az.plot_posterior(data, var_names=filtered_variables)  # Replace 'variable_name'
    plt.tight_layout
    plt.show()

    az.plot_trace(data, var_names=filtered_variables)  # Replace 'variable_name'
    plt.tight_layout
    plt.show()


except FileNotFoundError:
    print("Error: The specified file does not exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
