import os
import json
import yaml
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import RoothPath

# =====================================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================================

COLOR_ST = '#1f77b4'    
COLOR_MS = '#ff7f0e'    
COLOR_SIM = '#2ca02c'   
COLOR_TF = '#9467bd'    
COLOR_LB = '#2ca02c'    
COLOR_UB = '#d62728'    

# "Radius": r'$R$' come richiesto
PARAM_AXIS_LABEL = {
    'Alpha': r'$\alpha$',
    'Radius': r'$R$',  
    'Lookahead': r'$\lambda$',
    'N_Agents': r'$N_{\mathrm{agents}}$',
    'N_Tasks': r'$N_{\mathrm{tasks}}$',
    'alpha': r'$\alpha$',
    'radius': r'$R$',
    'lookahead': r'$\lambda$',
    'Laziness': r'$\lambda$',
}

SCENARIOS = {
    "Scenario_1": {"alpha": 0.1,  "radius": 3, "lookahead": 5},
    "Scenario_2": {"alpha": 1.0,  "radius": 3, "lookahead": 1},
    "Scenario_3": {"alpha": 20.0, "radius": 5, "lookahead": 3},
}

DEFAULT_N_TASKS_FOR_PARAM_SWEEP = 20
SCALE_DEFAULT_N_TASKS = 20

# =====================================================================================
# PLOTTING HELPERS
# =====================================================================================

def clip_series(values, lb=None, ub=None):
    v = np.array(values, dtype=float)
    if lb is not None:
        lb_arr = np.full_like(v, float(lb)) if np.isscalar(lb) else np.array(lb, dtype=float)
        mask = ~np.isnan(v) & ~np.isnan(lb_arr)
        v[mask] = np.maximum(v[mask], lb_arr[mask])
    if ub is not None:
        ub_arr = np.full_like(v, float(ub)) if np.isscalar(ub) else np.array(ub, dtype=float)
        mask = ~np.isnan(v) & ~np.isnan(ub_arr)
        v[mask] = np.minimum(v[mask], ub_arr[mask])
    return v

def _style_axes(ax):
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def make_plot(x, y, yerr, out_path, *, color, marker, ylabel, param_name,
              metric_title, base_text, lb=None, ub=None, lb_is_series=False,
              ub_is_series=False, clip_bounds=True, clip_ub=True, force_ylim_zero=False):
    
    fig, ax = plt.subplots(figsize=(8, 6))
    y_plot = clip_series(y, lb=lb if clip_bounds else None, ub=ub if (clip_bounds and clip_ub) else None)

    ax.errorbar(x, y_plot, yerr=yerr, fmt=f'-{marker}', color=color, ecolor='gray',
                elinewidth=1.3, capsize=4, capthick=1.3, markersize=7, linewidth=1.8,
                label='T-A* (Avg)', zorder=3)

    if lb is not None:
        if lb_is_series:
            ax.plot(x, lb, '--', color=COLOR_LB, linewidth=1.8, label='Lower Bound (No Movable Obs.)', zorder=2)
        else:
            ax.axhline(lb, color=COLOR_LB, linestyle='--', linewidth=1.8, label='Lower Bound (No Movable Obs.)', zorder=2)
    if ub is not None:
        if ub_is_series:
            ax.plot(x, ub, '--', color=COLOR_UB, linewidth=1.8, label='Upper Bound (Static Obs.)', zorder=2)
        else:
            ax.axhline(ub, color=COLOR_UB, linestyle='--', linewidth=1.8, label='Upper Bound (Static Obs.)', zorder=2)

    ax.set_xlabel(PARAM_AXIS_LABEL.get(param_name, param_name), fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(metric_title, fontsize=14, fontweight='bold', pad=12)
    
    if base_text:
        ax.text(0.02, 0.97, base_text, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    if force_ylim_zero:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top)
        
    _style_axes(ax)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def base_params_text(fixed_params, exclude_key):
    exclude_key = {'laziness': 'lookahead'}.get(exclude_key.lower(), exclude_key.lower())
    lines = []
    for k, v in fixed_params.items():
        if k.lower() != exclude_key:
            lines.append(f"{PARAM_AXIS_LABEL.get(k, k)} = {v}")
    return "Fixed:\n" + "\n".join(lines) if lines else ""

# =====================================================================================
# MAIN
# =====================================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" STANDALONE PLOT GENERATOR (FROM EXISTING JSON)")
    print("=" * 60)

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as f:
        config = json.load(f)

    map_yaml_name = config['input_name']
    map_name_clean = map_yaml_name.replace('.yaml', '')
    map_path = os.path.join(RoothPath.get_root(), config['input_path'], map_yaml_name)
    base_plots_dir = os.path.join(RoothPath.get_root(), 'plots_final', map_name_clean)

    with open(map_path, 'r') as f:
        map_param = yaml.safe_load(f)
        
    non_task_endpoints = map_param['map']['non_task_endpoints']
    max_possible_agents = len(non_task_endpoints)
    default_agents = max(1, max_possible_agents // 2)
    original_map_agents_len = len(map_param.get('agents', []))

    print(f"Map: {map_yaml_name}")
    print(f"Reading plots from directory: {base_plots_dir}\n")

    for scen_name, base_params in SCENARIOS.items():
        scen_dir = os.path.join(base_plots_dir, scen_name)
        json_path = os.path.join(scen_dir, 'scenario_results.json')
        
        if not os.path.exists(json_path):
            print(f"[!] File not found: {json_path}. Skipping {scen_name}.")
            continue
            
        print(f">>> Generating plots for {scen_name} <<<")
        
        with open(json_path, 'r') as f:
            master_results = json.load(f)
            
        PARAM_BASELINES = master_results.get('baselines_params_sweep', {})
        SCALE_BASELINES = master_results.get('baselines_scale_sweep', {})

        # -- 3a. Alpha / Radius / Laziness sweeps
        for param_name in ["Alpha", "Radius", "Laziness"]:
            if param_name not in master_results: continue
                
            entry = master_results[param_name]
            sweep_vals = entry['values']
            fixed_display_params = {**base_params, 'N_Tasks': DEFAULT_N_TASKS_FOR_PARAM_SWEEP, 'N_Agents': original_map_agents_len}
            base_text = base_params_text(fixed_display_params, param_name)
            
            lb_ms_v = PARAM_BASELINES.get('LB', {}).get('makespan', None)
            ub_ms_v = PARAM_BASELINES.get('UB', {}).get('makespan', None)
            lb_st_v = PARAM_BASELINES.get('LB', {}).get('service_time', None)
            ub_st_v = PARAM_BASELINES.get('UB', {}).get('service_time', None)
            # NUOVO: Lettura baseline del Sim Time
            lb_sim_v = PARAM_BASELINES.get('LB', {}).get('sim_time', None)
            ub_sim_v = PARAM_BASELINES.get('UB', {}).get('sim_time', None)

            make_plot(sweep_vals, entry['avg_service_time'], entry['std_service_time'],
                      os.path.join(scen_dir, f'Plot_ServiceTime_{param_name}.png'),
                      color=COLOR_ST, marker='o', ylabel='Avg. Delivery Time (timesteps)',
                      param_name=param_name, metric_title='Delivery Time (Pickup -> Delivery)', base_text=base_text,
                      lb=lb_st_v, ub=ub_st_v)

            make_plot(sweep_vals, entry['avg_makespan'], entry['std_makespan'],
                      os.path.join(scen_dir, f'Plot_Makespan_{param_name}.png'),
                      color=COLOR_MS, marker='s', ylabel='Avg. Makespan (timesteps)',
                      param_name=param_name, metric_title='Makespan', base_text=base_text,
                      lb=lb_ms_v, ub=ub_ms_v)

            # NUOVO: Passiamo lb_sim_v e ub_sim_v per disegnare i bound, Ylabel = "Execution Time (s)"
            make_plot(sweep_vals, entry['avg_sim_time'], entry['std_sim_time'],
                      os.path.join(scen_dir, f'Plot_SimTime_{param_name}.png'),
                      color=COLOR_SIM, marker='^', ylabel='Execution Time (s)',
                      param_name=param_name, metric_title='Execution Time', base_text=base_text,
                      lb=lb_sim_v, ub=ub_sim_v, force_ylim_zero=True, clip_ub=False)

            make_plot(sweep_vals, entry['avg_terraforming_cost'], entry['std_terraforming_cost'],
                      os.path.join(scen_dir, f'Plot_TerraformingCost_{param_name}.png'),
                      color=COLOR_TF, marker='d', ylabel=r'Terraforming Cost ($\alpha \cdot P$)',
                      param_name=param_name, metric_title='Terraforming Cost', base_text=base_text,
                      force_ylim_zero=True)

        # -- 3b. N_Agents / N_Tasks sweeps
        for param_name in ["N_Agents", "N_Tasks"]:
            if param_name not in master_results: continue
                
            entry = master_results[param_name]
            sweep_vals = entry['values']
            base_text = base_params_text({**base_params, ('N_Tasks' if param_name == 'N_Agents' else 'N_Agents'): (SCALE_DEFAULT_N_TASKS if param_name == 'N_Agents' else default_agents)}, param_name)
                
            lb_ms_arr = SCALE_BASELINES.get(param_name, {}).get('lb_ms', None)
            ub_ms_arr = SCALE_BASELINES.get(param_name, {}).get('ub_ms', None)
            lb_st_arr = SCALE_BASELINES.get(param_name, {}).get('lb_st', None)
            ub_st_arr = SCALE_BASELINES.get(param_name, {}).get('ub_st', None)
            # NUOVO: Lettura baseline array del Sim Time
            lb_sim_arr = SCALE_BASELINES.get(param_name, {}).get('lb_sim', None)
            ub_sim_arr = SCALE_BASELINES.get(param_name, {}).get('ub_sim', None)

            make_plot(sweep_vals, entry['avg_service_time'], entry['std_service_time'],
                      os.path.join(scen_dir, f'Plot_ServiceTime_{param_name}.png'),
                      color=COLOR_ST, marker='o', ylabel='Avg. Delivery Time (timesteps)',
                      param_name=param_name, metric_title='Delivery Time (Pickup -> Delivery)', base_text=base_text,
                      lb=lb_st_arr, ub=ub_st_arr, lb_is_series=True, ub_is_series=True)

            make_plot(sweep_vals, entry['avg_makespan'], entry['std_makespan'],
                      os.path.join(scen_dir, f'Plot_Makespan_{param_name}.png'),
                      color=COLOR_MS, marker='s', ylabel='Avg. Makespan (timesteps)',
                      param_name=param_name, metric_title='Makespan', base_text=base_text,
                      lb=lb_ms_arr, ub=ub_ms_arr, lb_is_series=True, ub_is_series=True)

            # NUOVO: Passiamo gli array lb_sim_arr e ub_sim_arr
            make_plot(sweep_vals, entry['avg_sim_time'], entry['std_sim_time'],
                      os.path.join(scen_dir, f'Plot_SimTime_{param_name}.png'),
                      color=COLOR_SIM, marker='^', ylabel='Execution Time (s)',
                      param_name=param_name, metric_title='Execution Time', base_text=base_text,
                      lb=lb_sim_arr, ub=ub_sim_arr, lb_is_series=True, ub_is_series=True, force_ylim_zero=True, clip_ub=False)

            make_plot(sweep_vals, entry['avg_terraforming_cost'], entry['std_terraforming_cost'],
                      os.path.join(scen_dir, f'Plot_TerraformingCost_{param_name}.png'),
                      color=COLOR_TF, marker='d', ylabel=r'Terraforming Cost ($\alpha \cdot P$)',
                      param_name=param_name, metric_title='Terraforming Cost', base_text=base_text,
                      force_ylim_zero=True)

        print(f"[+] Plots updated successfully for {scen_name}")

    print("\n" + "=" * 60)
    print(" ALL PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60 + "\n")