import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import RoothPath

# =====================================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================================

# Font sizes (enlarged for the paper)
FS_TICKS = 16
FS_AXIS_LABEL = 18
FS_TITLE = 18
FS_LEGEND = 15

X_LABEL = r'Number of agents ($m$)'

# Map folder -> paper name, and the scenario used for the team-size sweep in the paper
# (Scenario 1 everywhere, Scenario 3 on M6, the only preset whose radius makes it solvable)
MAP_NAMES = {
    'cross_with_pocket': ('M1', 'Scenario_1'),
    'shared_junction': ('M2', 'Scenario_1'),
    'input_tunnel': ('M3', 'Scenario_1'),
    'four_rooms': ('M4', 'Scenario_1'),
    'input_cross_scaled': ('M5', 'Scenario_1'),
    'spine_warehouse': ('M6', 'Scenario_3'),
    'traditional_warehouse_cross': ('M7', 'Scenario_1'),
}
MARKERS = ['o', 's', '^', 'd', 'v', 'P', 'X']

# =====================================================================================
# DATA
# =====================================================================================

def get_median_makespan(entry, where):
    """Median makespan per sweep point, from the per-seed values that
    experiments_unified.py stores in entry['raw_data'][str(value)]['makespans'].
    Failed runs are already excluded there, as for the means."""
    if 'raw_data' not in entry:
        raise KeyError(f"No 'raw_data' in {where}. Available keys: {sorted(entry.keys())}")
    med = []
    for v in entry['values']:
        seeds = entry['raw_data'].get(str(v), {}).get('makespans', [])
        s = np.array(seeds, dtype=float)
        s = s[~np.isnan(s)]
        med.append(np.median(s) if s.size else np.nan)
    return np.array(med)

# =====================================================================================
# PLOTTING
# =====================================================================================

def _style_axes(ax):
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=FS_TICKS)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def make_all_maps_plot(curves, out_path):
    """One curve per map: median makespan against team size."""
    fig, ax = plt.subplots(figsize=(8, 6))
    all_x = set()
    for i, (name, x, y) in enumerate(curves):
        ax.plot(x, y, f'-{MARKERS[i % len(MARKERS)]}', linewidth=1.8, markersize=7, label=name)
        all_x.update(x)

    ax.set_xlabel(X_LABEL, fontsize=FS_AXIS_LABEL)
    ax.set_ylabel('Median Makespan (timesteps)', fontsize=FS_AXIS_LABEL)
    ax.set_title('Median Makespan', fontsize=FS_TITLE, fontweight='bold', pad=12)
    ax.set_xticks(sorted(all_x))

    _style_axes(ax)
    ax.legend(loc='best', fontsize=FS_LEGEND, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# =====================================================================================
# MAIN
# =====================================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" MEDIAN MAKESPAN, ALL MAPS (FROM EXISTING JSON)")
    print("=" * 60)

    root = RoothPath.get_root()
    plots_root = os.path.join(root, 'plots_final')   # input: same JSON as before
    out_root = os.path.join(root, 'plots_new')       # output: new folder
    os.makedirs(out_root, exist_ok=True)

    curves = []
    for folder, (paper_name, scen_name) in MAP_NAMES.items():
        json_path = os.path.join(plots_root, folder, scen_name, 'scenario_results.json')
        if not os.path.exists(json_path):
            print(f"[!] File not found: {json_path}. {paper_name} left out of the combined plot.")
            continue
        with open(json_path, 'r') as f:
            master_results = json.load(f)
        if 'N_Agents' not in master_results:
            print(f"[!] No N_Agents sweep in {json_path}. {paper_name} left out.")
            continue
        entry = master_results['N_Agents']
        curves.append((paper_name, entry['values'], get_median_makespan(entry, json_path)))

    if curves:
        out_path = os.path.join(out_root, 'Plot_MedianMakespan_N_Agents_all_maps.png')
        make_all_maps_plot(curves, out_path)
        print(f"[+] {out_path}")

    print("\n" + "=" * 60)
    print(" DONE")
    print("=" * 60 + "\n")