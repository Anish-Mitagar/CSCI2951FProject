# Battle Arena – Quick Start Guide

## Installation

**Prerequisites:** Python 3.8+

Install UV package manager:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

Set up your environment:

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
uv pip install -e .
```

## Running Simulations

Start with defaults:

```bash
arena-run
```

Customize your run:

```bash
arena-run --num-episodes 500 --policies aggressive scanner custom random
arena-run --list-policies
```

Each policy name creates one agent. 
Repeat a policy name to spawn multiple agents with that policy:

```bash
arena-run --policies random random random
```

## Creating Custom Policies

Edit `arena/policies/custom.py`:

```python
from arena.constants import Channel
from arena.types import Action, Observation, Reward
from .policy import Policy

class Custom(Policy):
    def get_action(self, observation: Observation) -> Action:
        # Return action 0-3
        pass

    def update(self, observation: Observation, action: Action,
               reward: Reward) -> None:
        # Process feedback
        pass

    def reset(self) -> None:
        # Reset state between episodes
        pass
```

## Adding New Policies

Create `arena/policies/myPolicy.py` and register it in `arena/policies/__init__.py`:

```python
from .myPolicy import MyPolicy

REGISTRY = {
    "my-policy": MyPolicy,
    # ... existing policies
}
```

Verify and run:

```bash
arena-run --list-policies
arena-run --policies my-policy custom random
```

## Understanding Observations

Every step, each agent gets an observation made up of 3 layers. 
Each layer is a grid that matches the arena's height and width.

Layer 0 (Map): This is what the agent can see. 
It shows the grid with walls and any enemies within the agent's vision radius. 
The agent's own position is visible here, along with nearby agents. 
Anything outside the vision range shows up blank.

Layer 1 (Pre-Combat): Snapshots of where agents were positioned before any fights started. 
It only displays the IDs of agents that got involved in combat that step.

Layer 2 (Post-Combat): Shows where agents ended up after the fight was resolved. 
Only agents currently engaged in combat appear here. 
Agents still alive after the fight stay visible at their current location.

Compare layers 1 and 2 together to figure out who fought each other and how it turned out.

## Policy Method Call Order

Understanding the sequence in which agent methods are called is key to implementing your solution. 
Below is the step-by-step flow during an episode:

- Episode begins
- `reset()` - Called once at start. Agent initializes.
- Loop for each step:
	1. `get_action(observation)` - Agent decides action based on observation + internal state
	2. Agent moves
	3. Combat happens
	4. `update(observation, action, reward)` - Agent learns and update internal state
- Episode ends

One important caveat to understand is that the observation in `get_action()` and `update()` are from different times. 
The `get_action(...)` observation shows the state before your action happens. 
The `update(...)` observation shows the state after all agents have moved and combat has resolved. 
This means you see the consequences of your action when `update(...)` is called.

## Combat System

Strengths are assigned at the beginning of the session and remain constant throughout the episodes. 
Strengths are a number in the range 1 to N, where N is the number of players. 
Players get a unique strength, no two players will get the same number.

Battles happen when two or more agents are at the same position.
This can happen in two ways:

1. Two agents move to the same spot
2. Two agents pass through each other (agent A moves to B's old position while B moves to A's old position).

When two agents collide, the one with higher strength wins.
Tho loser player is eliminated from the game until an new episode begins.

There is a special rule: the player with strength 0 beats the highest strength player. 
This prevents any agent from being unbeatable.

If 3+ agents collide at the same spot, then the strongest agent fights the sum of all others' strengths. 
If the strongest agent is stronger than the sum of all other agents' strengths, it wins. 
Otherwise, the other players win.

You get rewarded by two mechanisms:

- Surviving a step
- Defeating an enemy

## Rendering Results

Convert runs to GIF:

```bash
arena-render all sessions
arena-render session sessions/2025-11-25_14-47-11
arena-render episode sessions/2025-11-25_14-47-11/0
```

Options:

```bash
--fps N          # Frames per second (default: 2)
```

Example:

```bash
arena-render --fps 5 all sessions
```

## Dependencies

Add packages with UV:

```bash
uv add scikit-learn
```

This auto-updates `pyproject.toml` and `uv.lock`.
