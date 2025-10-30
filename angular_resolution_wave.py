#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Constants
C_SPEED = 299792458.0  # m/s
RAD2DEG = 180.0 / np.pi

# Plot configuration
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})
plt.rc('axes', titlesize=16, linewidth=1.5)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# Complete instrument database
# Each instrument has: calculation method, data, collecting area, label position, color
INSTRUMENTS = {
    'eMERLIN': {
        'method': 'calculated',
        'freq_ghz': [1.25, 25],
        'baselines_km': [10, 217],
        'area_m2': 8000,
        'label_pos': (0.24, 0.15),
        'label_color': 'white'
    },
    'JVLA': {
        'method': 'manual',
        'wavelength': [4.054, 4.054, 0.00666, 0.00666],
        'resolution': [14563840.26674269, 14.003, 0.02303, 23949.42621642],
        'area_m2': 13200,
        'label_pos': (4.054, 14),
        'label_color': 'white',
        'label_text': 'VLA'
    },
    'SKAMID1': {
        'method': 'manual',
        'wavelength': [0.857, 0.857, 0.0196, 0.0196],
        'resolution': [2769.22039182, 0.7184, 0.01643, 170.583976136],
        'area_m2': 33000,
        'label_pos': (0.857, 0.7184 + 0.06),
        'label_color': 'royalblue',
        'label_text': 'SKA1-MID'
    },
    'ASKAP': {
        'method': 'manual',
        'wavelength': [0.42827494, 0.42827494, 0.1665513655556, 0.1665513655556],
        'resolution': [4015.36956814, 14.7230217499, 5.72561956939, 1561.53260983],
        'area_m2': 4000,
        'label_pos': (0.42827494, 17.7230217499),
        'label_color': 'white'
    },
    'SKALOW1': {
        'method': 'manual',
        'wavelength': [6.0, 6.0, 0.857, 0.857],
        'resolution': [16061.4782726, 11.606, 1.658, 5889.20869994],
        'area_m2': 419000,
        'label_pos': (6, 11.8),
        'label_color': 'royalblue',
        'label_text': 'SKA1-LOW'
    },
    'EVN': {
        'method': 'manual',
        'wavelength': [0.3, 0.3, 0.007, 0.007],
        'resolution': [0.232629, 10.57/1000., 0.25/1000., 0.006],
        'area_m2': 27000,
        'label_pos': (0.3, 10.57/1000.),
        'label_color': 'white'
    },
    'LOFAR': {
        'method': 'manual',
        'wavelength': [20, 20, 1.5, 1.5],
        'resolution': [60666., 1.93437, 0.14507, 3791.],
        'area_m2': 67000,
        'label_pos': (2.3, 0.25),
        'label_color': 'white'
    },
    'NOEMA': {
        'method': 'calculated',
        'freq_ghz': [70, 276],
        'baselines_km': [0.015, 1.6],
        'area_m2': 2220,
        'label_pos': 'auto',
        'label_color': 'white'
    },
    'SMA': {
        'method': 'calculated',
        'freq_ghz': [180, 420],
        'baselines_km': [0.008, 0.509],
        'area_m2': 226,
        'label_pos': 'auto',
        'label_color': 'white'
    },
    'ALMA': {
        'method': 'calculated',
        'freq_ghz': [84, 950],
        'baselines_km': [0.012, 16],
        'area_m2': 6600,
        'label_pos': 'auto',
        'label_color': 'white'
    },
    'MeerKAT': {
        'method': 'calculated',
        'freq_ghz': [0.580, 1.670],
        'baselines_km': [0.029, 7.7],
        'area_m2': 9000,
        'label_pos': 'auto',
        'label_color': 'white'
    },
    'Spitzer': {
        'method': 'calculated',
        'freq_ghz': [1870, 83200],
        'baselines_km': [0.00001, 0.00086/2],
        'area_m2': 0.5,
        'label_pos': 'auto',
        'label_color': 'white'
    },
    'Herschel': {
        'method': 'calculated',
        'freq_ghz': [447, 5259],
        'baselines_km': [0.00001, 0.0035/2],
        'area_m2': 9.6,
        'label_pos': 'auto',
        'label_color': 'white'
    },
    'Hubble': {
        'method': 'calculated',
        'freq_ghz': [176348, 2606890],
        'baselines_km': [1e-10, 0.0024/2],
        'area_m2': 4.525,
        'label_pos': 'auto',
        'label_color': 'white'
    },
    'JWST': {
        'method': 'manual',
        'wavelength': [6e-7, 6e-7, 2.85e-5, 2.85e-5],
        'resolution': [100000., 0.01875136363, 0.89068977272, 100000.],
        'area_m2': 25.4,
        'label_pos': (2.85e-5 - 0.5e-5, 0.89068977272),
        'label_color': 'white'
    },
    'ELT': {
        'method': 'manual',
        'wavelength': [3.7e-7, 3.7e-7, 2.4e-6, 2.4e-6],
        'resolution': [100000., 0.00194193511, 0.01259633587, 100000.],
        'area_m2': 878,
        'label_pos': (2.4e-6 - 0.1e-6, 0.01259633587),
        'label_color': 'royalblue'
    },
}

# Baseline scale reference lines
BASELINE_SCALES = [
    {'wavelength': 2.4e-7, 'resolution': 0.004, 'label': '10 m'},
    {'wavelength': 2.4e-6, 'resolution': 0.004, 'label': '100 m'},
    {'wavelength': 2.4e-5, 'resolution': 0.004, 'label': '1 km'},
    {'wavelength': 2.4e-4, 'resolution': 0.004, 'label': '10 km'},
    {'wavelength': 2.4e-3, 'resolution': 0.004, 'label': '100 km'},
    {'wavelength': 2.4e-2, 'resolution': 0.004, 'label': '1000 km'},
    {'wavelength': 2.4e-1, 'resolution': 0.004, 'label': '10000 km'},
]


def calc_angular_resolution_polygon(freq_ghz, baselines_km):
    """Calculate polygon vertices for angular resolution plot.
    
    Formula: 1.22 * lambda / baseline
    where lambda = c / freq
    """
    freq_ghz = np.array(freq_ghz)
    baselines_km = np.array(baselines_km)
    
    # Wavelength in meters
    wavelength = C_SPEED / (freq_ghz * 1e9)
    
    # Angular resolution in arcseconds
    res_min = 1.22 * C_SPEED / (freq_ghz * 1e9) / (baselines_km[0] * 1000.) * RAD2DEG * 3600. / 2.0
    res_max = 1.22 * C_SPEED / (freq_ghz * 1e9) / (baselines_km[1] * 1000.) * RAD2DEG * 3600. / 2.0
    
    x = [wavelength[0], wavelength[0], wavelength[1], wavelength[1]]
    y = [res_min[0], res_max[0], res_max[1], res_min[1]]
    
    return np.array([x, y]).T


def create_instrument_polygon(instrument_data):
    """Create polygon for instrument based on calculation method."""
    if instrument_data['method'] == 'calculated':
        return calc_angular_resolution_polygon(
            instrument_data['freq_ghz'],
            instrument_data['baselines_km']
        )
    else:  # manual
        coords = list(zip(instrument_data['wavelength'], instrument_data['resolution']))
        return np.array(coords)


def create_grid_lines(ax, ax2):
    """Create diagonal grid lines representing constant baseline."""
    x_grid = 10.0 ** np.arange(-8, 4, 1)
    grey_level = '0.9'
    lw = 15
    
    for i in range(-1, 9):
        y_grid = (x_grid * 206265.) / (10.0 ** i)
        ax.loglog(x_grid, y_grid, color=grey_level, linestyle='-', 
                 linewidth=lw, zorder=0)
        ax2.loglog(x_grid, y_grid, color=grey_level, linestyle='', 
                  linewidth=lw, zorder=0)


def setup_axes_labels(ax, ax2, tempax):
    """Configure axis labels and formatting."""
    # Get current tick positions
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()
    
    # Y-axis labels (angular resolution)
    y_labels = {
        0.001: '1 mas',
        0.01: '10 mas',
        0.1: '0.1"',
        1.0: '1"',
        10.0: '10"'
    }
    ax.set_yticklabels([y_labels.get(y, '') for y in yticks])
    
    # Physical scale labels at z=0.06 (~250 Mpc), scale = 1.167 kpc/"
    scale_labels = {
        0.001: '0.012',
        0.01: '0.12',
        0.1: '1.2',
        1.0: '12',
        10.0: '120'
    }
    ax2.set_yticklabels([scale_labels.get(y, '') for y in yticks])
    
    # X-axis labels (wavelength): 10m (left) -> 0.1μm (right)
    wavelength_labels = {
        10.0: r'10 m',
        1.0: r'1 m',
        0.1: r'10 cm',
        0.01: r'1 cm',
        0.001: r'1 mm',
        0.0001: r'100 $\mu$m',
        0.00001: r'10 $\mu$m',
        0.000001: r'1 $\mu$m',
        0.0000001: r'0.1 $\mu$m'
    }
    ax_labels = [wavelength_labels.get(x, '') for x in xticks]
    ax.set_xticklabels(ax_labels)
    
    # Frequency labels: 30 MHz (left) -> 3000 THz (right)
    freq_labels = {
        10.0: r'30 MHz',
        1.0: r'300 MHz',
        0.1: r'3 GHz',
        0.01: r'30 GHz',
        0.001: r'300 GHz',
        0.0001: r'3 THz',
        0.00001: r'30 THz',
        0.000001: r'300 THz',
        0.0000001: r'3000 THz'
    }
    ax2_labels = [freq_labels.get(x, '') for x in xticks]
    ax2.set_xticklabels(ax2_labels)
    
    # Axis titles
    ax.set_ylabel('Angular resolution', fontsize=20)
    ax.set_xlabel('Wavelength', fontsize=20)
    ax2.set_xlabel(r'Frequency', fontsize=20)
    tempax.set_ylabel(r'Physical scale at 250 Mpc [kpc]', fontsize=20, rotation=270)
    tempax.yaxis.labelpad = 20


def add_instrument_labels(polygons, rot_angle=-48):
    """Add text labels for instruments."""
    for name, data in INSTRUMENTS.items():
        if data['label_pos'] == 'auto':
            poly = polygons[name]
            x, y = poly[0, 0], poly[1, 1]
        else:
            x, y = data['label_pos']
        
        label_text = data.get('label_text', name)
        plt.text(x, y, label_text, ha='left', rotation=rot_angle, 
                color=data['label_color'])


def add_baseline_scale_labels(rot_angle=-48):
    """Add baseline scale reference labels."""
    for scale in BASELINE_SCALES:
        plt.text(scale['wavelength'], scale['resolution'], scale['label'],
                ha='left', rotation=rot_angle, color='0.5', fontsize=12)


# Main plotting
fig, ax = plt.subplots(figsize=(16, 8))
tempax = ax.twinx()
ax2 = tempax.twiny()

# Create background grid
create_grid_lines(ax, ax2)

# Build polygons and collect data for all instruments
all_polygons = {}
all_patches = []
all_colors = []

for name, data in INSTRUMENTS.items():
    poly = create_instrument_polygon(data)
    all_polygons[name] = poly
    all_patches.append(Polygon(poly))
    all_colors.append(np.log10(data['area_m2']))

# Create patch collection
p = PatchCollection(all_patches, alpha=0.8, edgecolor='black', 
                   linewidth=2, cmap='gist_earth')
p.set_array(np.array(all_colors))
ax.add_collection(p)

# Add colorbar
cbar = fig.colorbar(p, ax=ax, pad=0.12)
cbar.ax.set_ylabel(r'log (Collecting area m$^{2}$)', 
                   rotation=270, fontsize=20, labelpad=25)

# Configure axes
ax.set_ylim(0.001, 50.0)
ax2.set_ylim(0.001, 50.0)
ax.set_xlim(10.000001, 0.0000001)
ax2.set_xlim(10.000001, 0.0000001)
ax.grid(which='major', color='0.2', linestyle='-', 
       linewidth=2, zorder=0, alpha=0.1)

setup_axes_labels(ax, ax2, tempax)
add_instrument_labels(all_polygons)
add_baseline_scale_labels()

plt.savefig('radio_resolution.png', bbox_inches='tight', dpi=150)
