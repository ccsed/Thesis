import os
import json
import yaml
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import RoothPath

# =====================================================================================
# CONFIGURATION & AESTHETICS
# =====================================================================================

ROLE_STYLE = {
    'parking':  {'color': '#cce5ff', 'edge': '#0066cc', 'label': 'Parking Endpoint'},
    'start':    {'color': '#ffe5cc', 'edge': '#cc7a00', 'label': 'Pickup'},
    'goal':     {'color': '#e5f5e0', 'edge': '#31a354', 'label': 'Delivery'},
}

CELL_IN = 0.20      # Pollici per cella (garantisce scala uniforme tra mappe diverse)
DPI = 300

# =====================================================================================
# DRAWING LOGIC
# =====================================================================================

def _draw_role_square(ax, x, y, roles):
    """Draw a unit cell at (x, y) [bottom-left corner] as a square.
    If multiple roles are present, the square is divided into geometric sections."""
    n = len(roles)
    if n == 0:
        return

    # 1 ruolo: riempie l'intero quadrato
    if n == 1:
        style = ROLE_STYLE[roles[0]]
        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=style['color'],
                                        edgecolor=style['edge'], linewidth=1.2, zorder=1))
    
    # 2 ruoli: divide a metà con una diagonale
    elif n == 2:
        s0, s1 = ROLE_STYLE[roles[0]], ROLE_STYLE[roles[1]]
        ax.add_patch(patches.Polygon([(x, y), (x+1, y), (x, y+1)], 
                                     facecolor=s0['color'], edgecolor=s0['edge'], zorder=1, linewidth=1))
        ax.add_patch(patches.Polygon([(x+1, y), (x+1, y+1), (x, y+1)], 
                                     facecolor=s1['color'], edgecolor=s1['edge'], zorder=1, linewidth=1))
    
    # 3 ruoli: split a "Y"
    elif n >= 3:
        s0, s1, s2 = ROLE_STYLE[roles[0]], ROLE_STYLE[roles[1]], ROLE_STYLE[roles[2]]
        cx, cy = x + 0.5, y + 0.5
        
        ax.add_patch(patches.Polygon([(x, y), (x+1, y), (cx, cy)], 
                                     facecolor=s0['color'], edgecolor=s0['edge'], zorder=1, linewidth=1))
        ax.add_patch(patches.Polygon([(x, y), (x, y+1), (x+0.5, y+1), (cx, cy)], 
                                     facecolor=s1['color'], edgecolor=s1['edge'], zorder=1, linewidth=1))
        ax.add_patch(patches.Polygon([(x+1, y), (x+1, y+1), (x+0.5, y+1), (cx, cy)], 
                                     facecolor=s2['color'], edgecolor=s2['edge'], zorder=1, linewidth=1))

    # Bordo definito per l'intera cella
    ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='none', edgecolor='#888888',
                                    linewidth=1.2, zorder=1.5))


def generate_map_preview(yaml_path, out_path_stem):
    with open(yaml_path, 'r') as f:
        map_data = yaml.safe_load(f)

    dims = map_data['map']['dimensions']
    W, H = dims[0], dims[1]
    static_obs = map_data['map'].get('obstacles', [])
    movable_obs = map_data['map'].get('movable_obstacles', [])
    parking_endpoints = map_data['map'].get('non_task_endpoints', [])
    start_locs = map_data['map'].get('start_locations', [])
    goal_locs = map_data['map'].get('goal_locations', [])
    agents = map_data.get('agents', [])

    # Costruisci ruoli per cella
    cell_roles = defaultdict(list)
    for ep in parking_endpoints:
        if 'parking' not in cell_roles[tuple(ep)]: cell_roles[tuple(ep)].append('parking')
    for ep in start_locs:
        if 'start' not in cell_roles[tuple(ep)]: cell_roles[tuple(ep)].append('start')
    for ep in goal_locs:
        if 'goal' not in cell_roles[tuple(ep)]: cell_roles[tuple(ep)].append('goal')

    # Dimensione in pollici calcolata tramite CELL_IN
    fig, ax = plt.subplots(figsize=(W * CELL_IN, H * CELL_IN))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    
    # Tick su ogni singola cella per disegnare correttamente la griglia
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    
    # Rimuove numeri e trattini dagli assi
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Griglia interna a celle come nell'originale
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, zorder=0)

    def draw_square(coords, color, edgecolor='black', hatch=None, zorder=1):
        ax.add_patch(patches.Rectangle((coords[0], coords[1]), 1, 1, facecolor=color, edgecolor=edgecolor,
                                        hatch=hatch, zorder=zorder))

    # Disegna Endpoints
    for (x, y), roles in cell_roles.items():
        _draw_role_square(ax, x, y, roles)

    # Disegna Ostacoli
    for obs in static_obs:
        draw_square(obs, '#333333', zorder=2)

    for mobs in movable_obs:
        draw_square(mobs, '#ff9900', edgecolor='#cc0000', hatch='///', zorder=3)

    # Disegna Agenti con testo
    for agent in agents:
        start = agent['start']
        cx, cy = start[0] + 0.5, start[1] + 0.5
        ax.add_patch(patches.Circle((cx, cy), 0.35, facecolor='#0033cc', edgecolor='black', zorder=4))
        label = agent['name'].replace('agent', 'A')
        # Il testo si ridimensiona leggermente se necessario, ma base a 7pt per adattarsi alla cella
        ax.text(cx, cy, label, color='white', ha='center', va='center', fontsize=7, fontweight='bold', zorder=5)

    # Nessun titolo, nessuna legenda

    # Salvataggio
    fig.savefig(f'{out_path_stem}.png', dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{out_path_stem}.pdf', dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def render_legend(out_path_stem):
    """Crea e salva una legenda a sé stante."""
    handles = [
        patches.Patch(facecolor='#333333', edgecolor='black', label='Static Obstacle'),
        patches.Patch(facecolor='#ff9900', edgecolor='#cc0000', hatch='///', label='Movable Obstacle'),
        patches.Patch(facecolor=ROLE_STYLE['parking']['color'], edgecolor=ROLE_STYLE['parking']['edge'], label=ROLE_STYLE['parking']['label']),
        patches.Patch(facecolor=ROLE_STYLE['start']['color'], edgecolor=ROLE_STYLE['start']['edge'], label=ROLE_STYLE['start']['label']),
        patches.Patch(facecolor=ROLE_STYLE['goal']['color'], edgecolor=ROLE_STYLE['goal']['edge'], label=ROLE_STYLE['goal']['label']),
        Line2D([0], [0], marker='o', color='w', label='Agent', markerfacecolor='#0033cc', markersize=12, markeredgecolor='black'),
    ]
    
    fig = plt.figure(figsize=(7.0, 0.4))
    fig.legend(handles=handles, loc='center', ncol=6, frameon=False, fontsize=9,
               handlelength=1.4, columnspacing=1.2)
    
    fig.savefig(f'{out_path_stem}.png', dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{out_path_stem}.pdf', dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

# =====================================================================================
# MAIN
# =====================================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" UNIFORM MAP PREVIEW GENERATOR (SQUARES & GRID) ")
    print("=" * 60)

    # Legge config.json usando gli stessi identici percorsi
    config_path = os.path.join(RoothPath.get_root(), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    map_yaml_name = config['input_name']
    map_name_clean = map_yaml_name.replace('.yaml', '')
    map_path = os.path.join(RoothPath.get_root(), config['input_path'], map_yaml_name)

    base_plots_dir = os.path.join(RoothPath.get_root(), 'plots_final')
    map_plots_dir = os.path.join(base_plots_dir, map_name_clean)
    
    os.makedirs(map_plots_dir, exist_ok=True)

    map_img_stem = os.path.join(map_plots_dir, 'map_preview')
    legend_img_stem = os.path.join(base_plots_dir, 'map_legend')
    
    print(f"Reading Map: {map_yaml_name}")
    print(f"Generating preview...")

    generate_map_preview(map_path, map_img_stem)
    print(f"[✓] Map Preview Saved at: {map_img_stem}.(png/pdf)")
    
    render_legend(legend_img_stem)
    print(f"[✓] Legend Saved at: {legend_img_stem}.(png/pdf)\n")