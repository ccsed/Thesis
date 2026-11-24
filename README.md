# Multi-Agent Pickup and Delivery in Configurable Environments

This repository contains the implementation of Terraforming A* (T-A*) and of the simulator used to evaluate it, accompanying the AAMAS 2027 submission *Multi-Agent Pickup and Delivery in Configurable Environments*.

In Configurable MAPD (C-MAPD), agents serve a stream of pickup and delivery tasks among movable obstacles. An agent can command a movable obstacle to move out of its way: each obstacle step costs α, and an obstacle can never move farther than a radius R from its initial position. Tasks are assigned with Token Passing; T-A* plans each agent's path and decides which obstacles to displace, issuing displacement commands up to λ steps ahead of the agent.

## Requirements

Python 3 (tested with Python 3.13). Install the dependencies with:

```
pip install -r requirements.txt
```

Saving the animation as a video also requires `ffmpeg` on the system path.

## Running a simulation

1. Choose a map by setting `input_name` in `config.json` to one of the files in `Environments/`.
2. Run:

```
python demo.py -alpha 0.1 -radius 3 -laziness 5
```

| Argument | Meaning | Default |
|---|---|---|
| `-alpha` | Cost α of one obstacle step, in units of agent timesteps | 100 |
| `-radius` | Displacement radius R | 1 |
| `-laziness` | Lookahead λ: how many steps ahead of the agent T-A* looks for obstacles to displace | 1 |
| `-a_star_max_iter` | Maximum number of states expanded by a single T-A* query | 5000 |
| `-no_vis` | Skip the animation at the end of the run | off |
| `-slow_factor` | Slows down the animation | 1 |

The options `-k`, `-p`, `-pd` and `-p_iter` belong to the delay-robust variants of Token Passing that the simulator was originally built for. They are not used by C-MAPD and should be left at their defaults.

At the end of a run, `output.yaml` contains the agents' and obstacles' paths (`schedule`), the completion time of each task (`completed_tasks_times`), and the total cost, i.e. the agents' travel time plus α times the number of obstacle steps (`cost`).

## Map format

Each file in `Environments/` is a YAML file with:

- `map.dimensions`: grid width and height;
- `map.obstacles`: static obstacles;
- `map.movable_obstacles`: initial cells of the movable obstacles;
- `map.non_task_endpoints`: cells where free agents can rest;
- `map.start_locations`, `map.goal_locations`: cells from which pickup and delivery locations are drawn;
- `agents`: name and start cell of each agent;
- `n_tasks`, `task_freq`: number of tasks and arrival rate (tasks per timestep) of the Poisson process that generates them;
- `n_delays_per_agent`: kept at 0 (delays are not part of C-MAPD).

The maps used in the paper are:

| Paper | File |
|---|---|
| M1 | `cross_with_pocket.yaml` |
| M2 | `shared_junction.yaml` |
| M3 | `input_tunnel.yaml` |
| M4 | `four_rooms.yaml` |
| M5 | `input_cross_scaled.yaml` |
| M6 | `spine_warehouse.yaml` |
| M7 | `traditional_warehouse_cross.yaml` |

## Code structure

- `demo.py`: runs one simulation on the map selected in `config.json`.
- `Simulation/TP_with_recovery.py`: Token Passing with movable obstacles (task assignment, token, calls to the low-level planner).
- `Simulation/CBS/a_star.py`: the T-A* search.
- `Simulation/CBS/cbs.py`: the search environment (state expansion, displacement delegations, lookahead, conflict checks).
- `Simulation/simulation_new_recovery.py`: the simulator that executes agent and obstacle paths step by step.
- `Simulation/tasks_and_delays_maker.py`: random task stream generation.
- `Utils/Visualization/visualize.py`: animation of a run.

## License

MIT, see `LICENSE`. The simulator builds on an existing open-source implementation of Token Passing and CBS; its attribution is restored after the review period.
