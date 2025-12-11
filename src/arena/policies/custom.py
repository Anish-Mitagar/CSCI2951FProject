"""
Custom Policy Implementation for Battle Arena
==============================================

This policy implements a POMDP-inspired belief tracking approach for the Battle Arena
sequential decision-making project. It maintains probability distributions over
opponent strengths and uses Bayesian updates based on observed battle outcomes.

IMPORTANT CLARIFICATION:
- Strengths range from 1 to N (where N = number of players)
- Each player has a UNIQUE strength (no two players share the same strength)
- The special rule: strength 1 (weakest) beats ONLY the ORIGINAL strength N holder
- If the original strength-N player dies, the special rule becomes inactive

Theoretical Foundation (from CS2951F course materials):
-------------------------------------------------------
- State Uncertainty (POMDP, Ch. 19): Unknown player strengths, fog of war
- Interaction Uncertainty (Ch. 24-26): Other agents' policies unknown
- Exploration-Exploitation (Ch. 15): Thompson sampling for action selection
- Bayesian belief updates based on observed battle outcomes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

from arena.constants import Channel, Direction
from arena.types import Action, Observation, Reward
from .policy import Policy


@dataclass
class PlayerInfo:
    """Information tracked about each player across episodes."""
    player_id: int
    last_known_position: Optional[Tuple[int, int]] = None
    last_seen_step: int = -1
    is_alive: bool = True
    # Battle outcome tracking for strength inference
    defeated_by: Set[int] = field(default_factory=set)
    has_defeated: Set[int] = field(default_factory=set)
    # Strength bounds [lower, upper] inclusive, 1-indexed
    strength_lower: int = 1
    strength_upper: int = -1  # Set to N when known


@dataclass
class StrengthBelief:
    """
    Categorical probability distribution over possible strength values.
    
    belief[i] = P(strength = i+1) where i is 0-indexed
    Strengths are 1-indexed: [1, 2, ..., N]
    
    This follows the discrete belief state representation from Ch. 19 (Beliefs).
    """
    n_players: int
    belief: np.ndarray = None
    
    def __post_init__(self):
        if self.belief is None:
            # Uniform prior over all strengths [1, N]
            self.belief = np.ones(self.n_players) / self.n_players
    
    def copy(self) -> 'StrengthBelief':
        """Create a deep copy of this belief."""
        new_belief = StrengthBelief(self.n_players)
        new_belief.belief = self.belief.copy()
        return new_belief
    
    def update_won_against(self, loser_belief: 'StrengthBelief',
                           is_special_case: bool = False):
        """
        Bayesian update given this player won against another.
        
        Standard case: winner's strength > loser's strength
        Special case: winner has strength 1, loser had original strength N
        """
        if is_special_case:
            # Winner definitely has strength 1
            self.belief = np.zeros(self.n_players)
            self.belief[0] = 1.0
            return
        
        # P(my_strength = s | I won) ∝ P(loser_strength < s) × P(my_strength = s)
        new_belief = np.zeros(self.n_players)
        for s in range(self.n_players):
            # P(loser has strength < s+1) = sum of loser_belief[0:s]
            prob_loser_weaker = np.sum(loser_belief.belief[:s]) if s > 0 else 0.0
            new_belief[s] = prob_loser_weaker * self.belief[s]
        
        total = np.sum(new_belief)
        if total > 1e-10:
            self.belief = new_belief / total
    
    def update_lost_against(self, winner_belief: 'StrengthBelief',
                            is_special_case: bool = False):
        """
        Bayesian update given this player lost against another.
        
        Standard case: loser's strength < winner's strength
        Special case: loser had original strength N, winner has strength 1
        """
        if is_special_case:
            # Loser definitely has strength N
            self.belief = np.zeros(self.n_players)
            self.belief[-1] = 1.0
            return
        
        # P(my_strength = s | I lost) ∝ P(winner_strength > s) × P(my_strength = s)
        new_belief = np.zeros(self.n_players)
        for s in range(self.n_players):
            # P(winner has strength > s+1) = sum of winner_belief[s+1:]
            prob_winner_stronger = np.sum(winner_belief.belief[s+1:]) if s < self.n_players - 1 else 0.0
            new_belief[s] = prob_winner_stronger * self.belief[s]
        
        total = np.sum(new_belief)
        if total > 1e-10:
            self.belief = new_belief / total
    
    def apply_bounds(self, lower: int, upper: int):
        """Zero out probabilities outside [lower, upper] bounds (1-indexed)."""
        for s in range(self.n_players):
            strength = s + 1  # Convert to 1-indexed
            if strength < lower or strength > upper:
                self.belief[s] = 0.0
        
        total = np.sum(self.belief)
        if total > 1e-10:
            self.belief = self.belief / total
    
    def expected_strength(self) -> float:
        """Expected value of strength."""
        strengths = np.arange(1, self.n_players + 1)
        return float(np.sum(strengths * self.belief))
    
    def variance(self) -> float:
        """Variance of strength belief (measure of uncertainty)."""
        strengths = np.arange(1, self.n_players + 1)
        mean = self.expected_strength()
        return float(np.sum(self.belief * (strengths - mean) ** 2))
    
    def entropy(self) -> float:
        """Shannon entropy of belief (measure of uncertainty)."""
        # Avoid log(0)
        safe_belief = np.clip(self.belief, 1e-10, 1.0)
        return float(-np.sum(self.belief * np.log(safe_belief)))
    
    def prob_is_strength(self, strength: int) -> float:
        """Probability this player has the given strength (1-indexed)."""
        if 1 <= strength <= self.n_players:
            return float(self.belief[strength - 1])
        return 0.0
    
    def sample_strength(self) -> int:
        """Thompson sampling: draw a strength from the belief distribution."""
        total = np.sum(self.belief)
        if total < 1e-10:
            return np.random.randint(1, self.n_players + 1)
        probs = self.belief / total
        return int(np.random.choice(np.arange(1, self.n_players + 1), p=probs))
    
    def prob_stronger_than(self, other: 'StrengthBelief') -> float:
        """P(self's strength > other's strength), ignoring special rule."""
        prob = 0.0
        for my_s in range(self.n_players):
            for their_s in range(my_s):  # their_s < my_s
                prob += self.belief[my_s] * other.belief[their_s]
        return prob


class Custom(Policy):
    """
    Belief-tracking POMDP policy for Battle Arena.
    
    Key Features:
    - Maintains probability distributions over all players' strengths
    - Bayesian updates from observed battle outcomes
    - Tracks the "original strongest" for special rule handling
    - Three-phase strategy: cautious exploration → tactical engagement → aggressive endgame
    
    Persistent Memory:
    - Beliefs and battle history persist across episodes (within a session)
    - Strengths are constant per session, so learning accumulates
    """
    
    # Class-level persistent memory (survives across episodes within session)
    _persistent_memory: Dict = None
    _episode_count: int = 0
    
    def __init__(self, identifier: int):
        """
        Initialize the policy.
        
        Args:
            identifier: Our player ID (provided by the arena)
        """
        super().__init__(identifier)
        
        # Episode-specific state
        self.step_count = 0
        self.grid_height: int = 0
        self.grid_width: int = 0
        self.n_players: int = 0
        self.my_position: Optional[Tuple[int, int]] = None
        
        # Initialize persistent memory on first instantiation
        if Custom._persistent_memory is None:
            Custom._persistent_memory = {
                'player_beliefs': {},       # player_id -> StrengthBelief
                'player_info': {},          # player_id -> PlayerInfo
                'battle_history': [],       # (winner_id, loser_id) tuples
                'n_players': 0,
                # Original strongest tracking (for special rule)
                'original_strongest_id': None,
                'original_strongest_alive': True,
                'original_strongest_confidence': 0.0,
            }
        
        # Episode-specific tracking
        self.players_in_fov: Set[int] = set()
        self.alive_players: Set[int] = set()
        self.dead_players: Set[int] = set()
        self.all_known_players: Set[int] = set()
        
        # Direction vectors (matching Direction enum)
        # UP=0: [-1, 0], RIGHT=1: [0, 1], DOWN=2: [1, 0], LEFT=3: [0, -1]
        self.action_to_delta = {
            0: (-1, 0),   # UP
            1: (0, 1),    # RIGHT
            2: (1, 0),    # DOWN
            3: (0, -1),   # LEFT
        }
    
    def _get_memory(self) -> Dict:
        """Access persistent memory."""
        return Custom._persistent_memory
    
    def _get_belief(self, player_id: int) -> StrengthBelief:
        """Get or create belief for a player."""
        mem = self._get_memory()
        n = max(self.n_players, mem['n_players'], 3)
        
        if player_id not in mem['player_beliefs']:
            mem['player_beliefs'][player_id] = StrengthBelief(n)
        elif mem['player_beliefs'][player_id].n_players < n:
            # Need to expand belief array if we discovered more players
            old_belief = mem['player_beliefs'][player_id]
            new_belief = StrengthBelief(n)
            # Redistribute probability
            new_belief.belief[:old_belief.n_players] = old_belief.belief
            new_belief.belief = new_belief.belief / np.sum(new_belief.belief)
            mem['player_beliefs'][player_id] = new_belief
        
        return mem['player_beliefs'][player_id]
    
    def _get_player_info(self, player_id: int) -> PlayerInfo:
        """Get or create info tracker for a player."""
        mem = self._get_memory()
        if player_id not in mem['player_info']:
            n = max(self.n_players, mem['n_players'], 3)
            mem['player_info'][player_id] = PlayerInfo(
                player_id=player_id,
                strength_upper=n
            )
        return mem['player_info'][player_id]
    
    def _get_my_belief(self) -> StrengthBelief:
        """Get belief about our own strength."""
        return self._get_belief(self.identifier)
    
    def _is_original_strongest_alive(self) -> bool:
        """Check if the original strength-N holder is believed to be alive."""
        return self._get_memory().get('original_strongest_alive', True)
    
    def _get_original_strongest_id(self) -> Optional[int]:
        """Get the ID of the player believed to hold original strength N."""
        return self._get_memory().get('original_strongest_id')
    
    def _parse_observation(self, observation: Observation) -> Dict:
        """
        Parse observation array into structured data.
        
        Observation shape: (height, width, channels) where:
        - Channel 0 (MAP): Visible grid with player IDs and walls
        - Channel 1 (PRE_COMBAT): Pre-battle positions of combatants
        - Channel 2 (POST_COMBAT): Post-battle positions of survivors
        
        Cell values:
        - 0: Empty
        - -1: Wall
        - >0: Player ID (1-indexed due to PLAYER_ID_OFFSET)
        """
        obs = np.array(observation)
        height, width, n_channels = obs.shape
        
        self.grid_height = height
        self.grid_width = width
        
        # Extract channels (last dimension)
        map_layer = obs[:, :, Channel.MAP.value]
        pre_combat = obs[:, :, Channel.PRE_COMBAT.value] if n_channels > 1 else None
        post_combat = obs[:, :, Channel.POST_COMBAT.value] if n_channels > 2 else None
        
        # Find visible players in map layer
        visible_players = {}
        my_position = None
        
        for row in range(height):
            for col in range(width):
                cell_value = map_layer[row, col]
                if cell_value > 0:  # Player ID
                    player_id = int(cell_value)
                    visible_players[player_id] = (row, col)
                    self.all_known_players.add(player_id)
                    
                    if player_id == self.identifier:
                        my_position = (row, col)
        
        # Parse battle outcomes from combat layers
        battles = []
        if pre_combat is not None and post_combat is not None:
            pre_positions = {}
            post_positions = {}
            
            for row in range(height):
                for col in range(width):
                    if pre_combat[row, col] > 0:
                        pid = int(pre_combat[row, col])
                        pre_positions[pid] = (row, col)
                        self.all_known_players.add(pid)
                    if post_combat[row, col] > 0:
                        pid = int(post_combat[row, col])
                        post_positions[pid] = (row, col)
            
            # Players in pre but not in post are losers (eliminated)
            losers = set(pre_positions.keys()) - set(post_positions.keys())
            winners = set(post_positions.keys())
            
            # Match losers to winners based on position
            for loser in losers:
                loser_pos = pre_positions.get(loser)
                matched = False
                for winner in winners:
                    winner_post = post_positions.get(winner)
                    winner_pre = pre_positions.get(winner)
                    # Battle occurred if they ended at same spot or crossed paths
                    if winner_post == loser_pos or winner_pre == loser_pos:
                        battles.append((winner, loser))
                        matched = True
                        break
                if not matched and len(winners) > 0:
                    # Fallback: assign to any winner at that location
                    battles.append((list(winners)[0], loser))
        
        return {
            'map': map_layer,
            'pre_combat': pre_combat,
            'post_combat': post_combat,
            'visible_players': visible_players,
            'my_position': my_position,
            'battles': battles,
            'height': height,
            'width': width,
        }
    
    def _update_original_strongest_tracking(self, winner_id: int, loser_id: int) -> bool:
        """
        Update tracking of who holds original strength N.
        
        Returns True if this battle was the special case (strength 1 beat strength N).
        """
        mem = self._get_memory()
        
        winner_belief = self._get_belief(winner_id)
        loser_belief = self._get_belief(loser_id)
        
        # Check if this looks like the special case
        prob_loser_is_N = loser_belief.prob_is_strength(self.n_players)
        prob_winner_is_1 = winner_belief.prob_is_strength(1)
        
        is_special_case = (prob_loser_is_N > 0.5 and prob_winner_is_1 > 0.5)
        
        if is_special_case:
            # Confirmed: loser was original strongest, winner was weakest
            mem['original_strongest_id'] = loser_id
            mem['original_strongest_alive'] = False
            mem['original_strongest_confidence'] = min(prob_loser_is_N, prob_winner_is_1)
            return True
        
        # Check if loser was the original strongest (dying in any battle)
        current_strongest = mem.get('original_strongest_id')
        
        if current_strongest == loser_id:
            mem['original_strongest_alive'] = False
        elif current_strongest is None and prob_loser_is_N > 0.7:
            # Infer this loser was probably the original strongest
            mem['original_strongest_id'] = loser_id
            mem['original_strongest_alive'] = False
            mem['original_strongest_confidence'] = prob_loser_is_N
        
        return False
    
    def _update_beliefs_from_battle(self, winner_id: int, loser_id: int):
        """
        Update strength beliefs based on observed battle outcome.
        
        Uses Bayesian inference:
        P(strength | battle_result) ∝ P(battle_result | strength) × P(strength)
        """
        mem = self._get_memory()
        mem['battle_history'].append((winner_id, loser_id))
        
        # Check for special case and update tracking
        is_special = self._update_original_strongest_tracking(winner_id, loser_id)
        
        # Get current beliefs
        winner_belief = self._get_belief(winner_id)
        loser_belief = self._get_belief(loser_id)
        
        # Get player info
        winner_info = self._get_player_info(winner_id)
        loser_info = self._get_player_info(loser_id)
        
        # Record battle relationships
        winner_info.has_defeated.add(loser_id)
        loser_info.defeated_by.add(winner_id)
        loser_info.is_alive = False
        
        # Perform Bayesian belief updates
        winner_belief.update_won_against(loser_belief, is_special)
        loser_belief.update_lost_against(winner_belief, is_special)
        
        # Update strength bounds (only for standard case)
        if not is_special:
            # Winner must be stronger than loser
            new_winner_lower = max(winner_info.strength_lower, loser_info.strength_lower + 1)
            new_loser_upper = min(loser_info.strength_upper, winner_info.strength_upper - 1)
            
            if new_winner_lower <= winner_info.strength_upper:
                winner_info.strength_lower = new_winner_lower
                winner_belief.apply_bounds(winner_info.strength_lower, winner_info.strength_upper)
            
            if new_loser_upper >= loser_info.strength_lower:
                loser_info.strength_upper = new_loser_upper
                loser_belief.apply_bounds(loser_info.strength_lower, loser_info.strength_upper)
    
    def _prob_can_defeat(self, my_belief: StrengthBelief,
                          opponent_belief: StrengthBelief,
                          opponent_id: int) -> float:
        """
        Compute probability we can defeat this opponent.
        
        Accounts for the special rule: strength 1 beats original strength N,
        but ONLY if the original strongest is still alive.
        """
        # Base probability: we're stronger
        prob_stronger = my_belief.prob_stronger_than(opponent_belief)
        
        # Special rule bonus (only if original strongest is alive)
        if self._is_original_strongest_alive():
            original_strongest = self._get_original_strongest_id()
            
            # Check if opponent might be the original strongest
            is_potential_strongest = (
                original_strongest == opponent_id or
                (original_strongest is None and 
                 opponent_belief.prob_is_strength(self.n_players) > 0.3)
            )
            
            if is_potential_strongest:
                # Add P(we're 1) × P(they're N)
                prob_special = (my_belief.prob_is_strength(1) * 
                               opponent_belief.prob_is_strength(self.n_players))
                return prob_stronger + prob_special
        
        return prob_stronger
    
    def _update_player_positions(self, visible_players: Dict[int, Tuple[int, int]]):
        """Update last known positions for visible players."""
        for player_id, position in visible_players.items():
            info = self._get_player_info(player_id)
            info.last_known_position = position
            info.last_seen_step = self.step_count
            if player_id != self.identifier:
                self.players_in_fov.add(player_id)
    
    def _count_alive_players(self) -> int:
        """
        Estimate number of players still alive, INCLUDING ourselves.
        
        Returns at least 2 (us + at least one opponent assumed).
        When this returns exactly 2, we know it's endgame (1v1).
        """
        mem = self._get_memory()
        
        # Count from player_info (players we've observed)
        alive_from_info = set()
        for pid, info in mem['player_info'].items():
            if info.is_alive:
                alive_from_info.add(pid)
        
        # Combine with episode tracking
        alive_from_episode = self.alive_players - self.dead_players
        
        # Union of both sources
        all_alive = alive_from_info.union(alive_from_episode)
        
        # CRITICAL: Always include ourselves - we're alive if this code is running!
        all_alive.add(self.identifier)
        
        # Remove any dead players (safety check)
        all_alive = all_alive - self.dead_players
        
        # Return count, minimum of 2 (us + assumed opponent)
        return max(len(all_alive), 2)
    
    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance between two grid positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _get_action_probabilities(self, parsed_obs: Dict) -> np.ndarray:
        """
        Compute action probabilities based on current beliefs and game state.
        
        Strategy Phases:
        1. >2 alive, FOV empty: Explore, bias away from believed-stronger opponents
        2. >2 alive, 1 opponent visible: Engage if favorable, else evade
        3. >2 alive, multiple visible: Evade to avoid multi-combat
        4. ≤2 alive (endgame): Aggressive pursuit
        """
        my_pos = parsed_obs.get('my_position')
        if my_pos is None:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        self.my_position = my_pos
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Visible opponents (excluding self)
        visible_opponents = {
            pid: pos for pid, pos in parsed_obs['visible_players'].items()
            if pid != self.identifier
        }
        
        n_alive = self._count_alive_players()
        fov_empty = len(visible_opponents) == 0
        my_belief = self._get_my_belief()
        
        if n_alive <= 2:
            # ENDGAME: Only 1 opponent left (n_alive=2 means us + 1 opponent)
            # Go all-in on aggressive pursuit
            probs = self._compute_aggressive_probs(my_pos, visible_opponents)
        elif fov_empty:
            # EXPLORATION: Bias away from believed-stronger opponents
            probs = self._compute_exploration_probs(my_pos, my_belief)
        elif len(visible_opponents) == 1:
            # SINGLE OPPONENT: Assess and possibly engage
            probs = self._compute_engagement_probs(my_pos, visible_opponents, my_belief)
        else:
            # MULTIPLE OPPONENTS: Evade to avoid multi-agent combat
            probs = self._compute_evasion_probs(my_pos, visible_opponents)
        
        # Ensure valid probability distribution with exploration floor
        probs = np.maximum(probs, 0.05)
        probs = probs / np.sum(probs)
        
        return probs
    
    def _compute_aggressive_probs(self, my_pos: Tuple[int, int],
                                   visible_opponents: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Endgame strategy: move toward any known opponent."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Find closest target
        target_pos = None
        min_dist = float('inf')
        
        # Check visible opponents first
        for pid, pos in visible_opponents.items():
            dist = self._euclidean_distance(my_pos, pos)
            if dist < min_dist:
                min_dist = dist
                target_pos = pos
        
        # Fall back to last known positions
        if target_pos is None:
            mem = self._get_memory()
            for pid, info in mem['player_info'].items():
                if info.is_alive and info.last_known_position and pid != self.identifier:
                    dist = self._euclidean_distance(my_pos, info.last_known_position)
                    if dist < min_dist:
                        min_dist = dist
                        target_pos = info.last_known_position
        
        if target_pos is None:
            return probs
        
        # Strongly bias toward target
        for action, (dr, dc) in self.action_to_delta.items():
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            new_dist = self._euclidean_distance(new_pos, target_pos)
            
            if new_dist < min_dist:
                probs[action] += 0.5
            elif new_dist > min_dist:
                probs[action] -= 0.1
        
        return probs
    
    def _compute_exploration_probs(self, my_pos: Tuple[int, int],
                                    my_belief: StrengthBelief) -> np.ndarray:
        """
        Exploration strategy when FOV is empty.
        
        Behavior:
        - Move AWAY from positions of opponents we believe are STRONGER
        - Move TOWARD positions of opponents we believe are WEAKER
        - Stochastic base to maintain exploration
        
        Uses last-known positions weighted by belief in relative strength.
        """
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        mem = self._get_memory()
        
        for pid, info in mem['player_info'].items():
            if not info.is_alive or pid == self.identifier:
                continue
            if info.last_known_position is None:
                continue
            
            opp_pos = info.last_known_position
            opp_belief = self._get_belief(pid)
            current_dist = self._euclidean_distance(my_pos, opp_pos)
            
            if current_dist < 0.1:
                continue
            
            # Probability we win against this opponent
            prob_win = self._prob_can_defeat(my_belief, opp_belief, pid)
            
            # Determine behavior based on relative strength belief
            # prob_win > 0.5: we're likely stronger → move TOWARD (hunt them)
            # prob_win < 0.5: we're likely weaker → move AWAY (avoid them)
            
            for action, (dr, dc) in self.action_to_delta.items():
                new_pos = (my_pos[0] + dr, my_pos[1] + dc)
                new_dist = self._euclidean_distance(new_pos, opp_pos)
                
                # Distance change: positive = moving away, negative = moving toward
                moves_toward = new_dist < current_dist
                moves_away = new_dist > current_dist
                
                if prob_win > 0.55:
                    # We're likely STRONGER - move toward to hunt
                    strength_factor = (prob_win - 0.5) * 2  # 0 to 1
                    if moves_toward:
                        probs[action] += 0.3 * strength_factor
                    elif moves_away:
                        probs[action] -= 0.15 * strength_factor
                        
                elif prob_win < 0.45:
                    # We're likely WEAKER - move away to avoid
                    weakness_factor = (0.5 - prob_win) * 2  # 0 to 1
                    if moves_away:
                        probs[action] += 0.3 * weakness_factor
                    elif moves_toward:
                        probs[action] -= 0.15 * weakness_factor
                
                # If prob_win is near 0.5, we're uncertain - don't bias strongly
        
        return probs
    
    def _compute_engagement_probs(self, my_pos: Tuple[int, int],
                                   visible_opponents: Dict[int, Tuple[int, int]],
                                   my_belief: StrengthBelief) -> np.ndarray:
        """Single opponent visible: engage if favorable or uncertain."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        opponent_id = list(visible_opponents.keys())[0]
        opponent_pos = visible_opponents[opponent_id]
        current_dist = self._euclidean_distance(my_pos, opponent_pos)
        
        opponent_belief = self._get_belief(opponent_id)
        prob_win = self._prob_can_defeat(my_belief, opponent_belief, opponent_id)
        
        # Also consider uncertainty - high entropy means we should explore
        max_entropy = np.log(self.n_players) if self.n_players > 1 else 1.0
        uncertainty = opponent_belief.entropy() / max_entropy
        
        # Engage if: prob_win >= 0.4 OR high uncertainty (explore to learn)
        should_engage = prob_win >= 0.4 or uncertainty > 0.7
        
        for action, (dr, dc) in self.action_to_delta.items():
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            new_dist = self._euclidean_distance(new_pos, opponent_pos)
            
            if should_engage:
                if new_dist < current_dist:
                    probs[action] += 0.4
                elif new_dist > current_dist:
                    probs[action] -= 0.15
            else:
                if new_dist > current_dist:
                    probs[action] += 0.4
                elif new_dist < current_dist:
                    probs[action] -= 0.15
        
        return probs
    
    def _compute_evasion_probs(self, my_pos: Tuple[int, int],
                                visible_opponents: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Multiple opponents visible: evade to avoid multi-agent combat."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Move away from centroid of all opponents
        positions = list(visible_opponents.values())
        centroid = (
            np.mean([p[0] for p in positions]),
            np.mean([p[1] for p in positions])
        )
        current_dist = self._euclidean_distance(my_pos, centroid)
        
        for action, (dr, dc) in self.action_to_delta.items():
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            new_dist = self._euclidean_distance(new_pos, centroid)
            
            if new_dist > current_dist:
                probs[action] += 0.5
            elif new_dist < current_dist:
                probs[action] -= 0.2
        
        return probs
    
    def get_action(self, observation: Observation) -> Action:
        """
        Select an action based on observation and beliefs.
        
        Called each step BEFORE movement/combat.
        """
        self.step_count += 1
        
        # Parse observation
        parsed = self._parse_observation(observation)
        self._update_player_positions(parsed['visible_players'])
        
        # Update n_players estimate
        n_known = len(self.all_known_players)
        if n_known > self.n_players:
            self.n_players = n_known
            mem = self._get_memory()
            mem['n_players'] = max(mem['n_players'], n_known)
        
        if self.n_players == 0:
            self.n_players = max(3, n_known)
        
        # Compute action probabilities
        probs = self._get_action_probabilities(parsed)
        
        # Sample action stochastically
        action = int(np.random.choice(4, p=probs))
        
        return action
    
    def update(self, observation: Observation, action: Action, reward: Reward) -> None:
        """
        Process feedback after action execution.
        
        Called each step AFTER movement/combat resolution.
        Key for updating beliefs based on battle outcomes.
        """
        parsed = self._parse_observation(observation)
        
        # Process observed battles
        for winner_id, loser_id in parsed['battles']:
            self._update_beliefs_from_battle(winner_id, loser_id)
            self.dead_players.add(loser_id)
            self.alive_players.discard(loser_id)
        
        # Update alive tracking
        for pid in parsed['visible_players'].keys():
            if pid not in self.dead_players:
                self.alive_players.add(pid)
        
        self._update_player_positions(parsed['visible_players'])
        
        if parsed['my_position'] is not None:
            self.my_position = parsed['my_position']
    
    def reset(self) -> None:
        """
        Reset for new episode.
        
        Persistent memory (beliefs, battle history) survives across episodes.
        Strengths are constant within a session, so accumulated learning persists.
        """
        Custom._episode_count += 1
        
        # Reset episode-specific state
        self.step_count = 0
        self.my_position = None
        self.players_in_fov = set()
        self.alive_players = set()
        self.dead_players = set()
        
        # Reset alive status for all players (new episode = everyone respawns)
        mem = self._get_memory()
        for pid, info in mem['player_info'].items():
            info.is_alive = True
            info.last_seen_step = -1
        
        # Original strongest is alive again in new episode
        if mem.get('original_strongest_id') is not None:
            mem['original_strongest_alive'] = True
        
        # Beliefs persist across episodes (key for learning!)