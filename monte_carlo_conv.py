#!/usr/bin/env python
"""
Monte Carlo simulation using convolution fit model.
Runs 3000 iterations and saves fit results for valid fits.
"""

import os
import uproot
import numpy as np
import fitle as fl
import vector

# Create output directory
os.makedirs('mc_results', exist_ok=True)

# Load MERR data
print("Loading MERR data...")
f = uproot.open("../data/MERR.root")
Dp_DTF_MERR = f['DecayTree']["Dp_DTF_MERR"].array()

# Load PHOTOS data for D+
print("Loading D+ PHOTOS data...")
Dp_tree = uproot.open("../data/mc/raw/Dp.root")["Truth"]["t"]

Dp_ssK_vec = vector.array({
    "px": np.asarray(Dp_tree["ssK_TRUEP_X"]),
    "py": np.asarray(Dp_tree["ssK_TRUEP_Y"]),
    "pz": np.asarray(Dp_tree["ssK_TRUEP_Z"]),
    "E": np.asarray(Dp_tree["ssK_TRUEP_E"])
})
Dp_osK_vec = vector.array({
    "px": np.asarray(Dp_tree["osK_TRUEP_X"]),
    "py": np.asarray(Dp_tree["osK_TRUEP_Y"]),
    "pz": np.asarray(Dp_tree["osK_TRUEP_Z"]),
    "E": np.asarray(Dp_tree["osK_TRUEP_E"])
})
Dp_sspi_vec = vector.array({
    "px": np.asarray(Dp_tree["sspi_TRUEP_X"]),
    "py": np.asarray(Dp_tree["sspi_TRUEP_Y"]),
    "pz": np.asarray(Dp_tree["sspi_TRUEP_Z"]),
    "E": np.asarray(Dp_tree["sspi_TRUEP_E"])
})

Dp_mother_mass_samples = (Dp_ssK_vec + Dp_osK_vec + Dp_sspi_vec).mass

# Load PHOTOS data for Ds
print("Loading Ds PHOTOS data...")
Ds_tree = uproot.open("../data/mc/raw/Ds.root")["Truth"]["t"]

Ds_ssK_vec = vector.array({
    "px": np.asarray(Ds_tree["ssK_TRUEP_X"]),
    "py": np.asarray(Ds_tree["ssK_TRUEP_Y"]),
    "pz": np.asarray(Ds_tree["ssK_TRUEP_Z"]),
    "E": np.asarray(Ds_tree["ssK_TRUEP_E"])
})
Ds_osK_vec = vector.array({
    "px": np.asarray(Ds_tree["osK_TRUEP_X"]),
    "py": np.asarray(Ds_tree["osK_TRUEP_Y"]),
    "pz": np.asarray(Ds_tree["osK_TRUEP_Z"]),
    "E": np.asarray(Ds_tree["osK_TRUEP_E"])
})
Ds_sspi_vec = vector.array({
    "px": np.asarray(Ds_tree["sspi_TRUEP_X"]),
    "py": np.asarray(Ds_tree["sspi_TRUEP_Y"]),
    "pz": np.asarray(Ds_tree["sspi_TRUEP_Z"]),
    "E": np.asarray(Ds_tree["sspi_TRUEP_E"])
})

Ds_mother_mass_samples = (Ds_ssK_vec + Ds_osK_vec + Ds_sspi_vec).mass

# Load precomputed histograms for convolution
print("Loading convolution histograms...")
Dp_x, Dp_c = np.load("../data/mc/histograms/Dp.npy")
Ds_x, Ds_c = np.load("../data/mc/histograms/Ds.npy")

# True mass values for convolution
Dp_mother_mass = 1869.65
Ds_mother_mass = 1968.33


def sample():
    """Generate a Monte Carlo sample with D+, Ds, and background."""
    raw_data_dp = (np.random.choice(Dp_mother_mass_samples, size=800000) +
                   np.multiply(np.random.choice(Dp_DTF_MERR, 800000), np.random.randn(800000)))
    raw_data_ds = (np.random.choice(Ds_mother_mass_samples, size=1000000) +
                   np.multiply(np.random.choice(Dp_DTF_MERR, 1000000), np.random.randn(1000000)))
    background = np.random.exponential(150, 60000) + 1840
    return np.concatenate([raw_data_dp, raw_data_ds, background])


def build_conv_model():
    """Build the convolution fit model."""
    mass2 = fl.Param(1970)('mass')
    mass1 = mass2 - fl.Param(100)('mass_diff')

    Dp = (fl.Param.positive(500000) * fl.convolve(Dp_x, Dp_c, Dp_mother_mass, mass1, fl.Param.positive(5)) +
          fl.Param.positive(500000) * fl.convolve(Dp_x, Dp_c, Dp_mother_mass, mass1, fl.Param.positive(10)))

    Ds = (fl.Param.positive(600000) * fl.convolve(Ds_x, Ds_c, Ds_mother_mass, mass2, fl.Param.positive(5)) +
          fl.Param.positive(600000) * fl.convolve(Ds_x, Ds_c, Ds_mother_mass, mass2, fl.Param.positive(10)))

    tail = fl.Param.positive(40000) * fl.exponential(tau=fl.Param.positive(100)) % (fl.INPUT - 1840)

    return Dp + Ds + tail


# Storage for results
mu_vals = []
mu_errs = []
diff_vals = []
diff_errs = []

N_ITERATIONS = 3000

print(f"Starting {N_ITERATIONS} Monte Carlo iterations...")

for i in range(N_ITERATIONS):
    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{N_ITERATIONS} - {len(mu_vals)} valid fits so far")

    # Generate sample
    observed = sample()

    # Build fresh model (parameters get reset)
    model = build_conv_model()

    # Fit
    cost = fl.Cost.chi2(observed, 200, range=(1840, 2040))
    fit = fl.fit(model | cost)

    # Store if valid
    if fit.minimizer.valid:
        mu_vals.append(fit.values['mass'])
        mu_errs.append(fit.errors['mass'])
        diff_vals.append(fit.values['mass_diff'])
        diff_errs.append(fit.errors['mass_diff'])

print(f"\nCompleted. {len(mu_vals)} valid fits out of {N_ITERATIONS} iterations.")

# Save results
np.array(mu_vals).tofile('mc_results/conv_mu_val')
np.array(mu_errs).tofile('mc_results/conv_mu_err')
np.array(diff_vals).tofile('mc_results/conv_diff_val')
np.array(diff_errs).tofile('mc_results/conv_diff_err')

print("Results saved to mc_results/conv_*")