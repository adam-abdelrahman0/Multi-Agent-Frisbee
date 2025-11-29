from pettingzoo import ParallelEnv
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class DynamicUltimateFrisbeeEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "ultimate_frisbee_v2"}

    def __init__(
        self,
        num_players_per_team: int = 2,
        use_regulation_field: bool = False,
        seed: int | None = None,
        render_mode: str | None = None,
        debug: bool = False,
        max_steps: int = 400,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.debug = debug
        self.num_players_per_team = num_players_per_team
        self.num_players = num_players_per_team * 2
        self.max_steps = max_steps
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        # Field size
        if use_regulation_field:
            self.endzone_depth = 20
            self.playing_length = 70
            self.playing_width = 40
            self.field_size = (self.playing_length + 2 * self.endzone_depth, self.playing_width)
        else:
            self.field_size = (10, 10)
            self.endzone_depth = 0
            self.playing_length = 10
            self.playing_width = 10

        # Agents
        self.agents = [f"team_{t}_player_{i}" for t in range(2) for i in range(num_players_per_team)]
        self.agent_order = list(self.agents)

        # Observation/action spaces
        obs_dim = self._obs_dim()
        self.observation_spaces = {a: spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32) for a in self.agents}
        self.action_spaces = {a: spaces.Discrete(5) for a in self.agents}

        # Gameplay parameters
        self.intercept_range = 1.0
        self.catch_range = 4.0  # Option 1: increased from 3.0
        self.defender_mark_distance = 2.0
        self.possession_stall_limit = 15

        # Rendering
        self._fig = None
        self._ax = None

        # State
        self.defensive_marks = {}
        self._pending_thrower = None
        self._prev_dist_to_goal = None
        self.prev_positions = {}

        self.reset()

    # ---------------------------------------
    # PettingZoo required wrappers
    # ---------------------------------------
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ---------------------------------------
    # Utility
    # ---------------------------------------
    def _team_of(self, agent):
        return int(agent.split("_")[1])

    def _obs_dim(self):
        t = self.num_players_per_team
        return 2 + 2 + 1 + 2 * (t - 1) + 2 * t

    def _norm(self, xy):
        w, h = self.field_size
        return np.array([xy[0] / (w - 1), xy[1] / (h - 1)], dtype=np.float32)

    # ---------------------------------------
    # Reset
    # ---------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed

        self.steps = 0
        field_w, field_h = self.field_size

        ys = np.round(np.linspace(0, field_h - 1, num=self.num_players_per_team)).astype(int)
        self.agent_positions = {}

        left_x = self.endzone_depth
        right_x = self.endzone_depth + self.playing_length

        for i, y in enumerate(ys):
            self.agent_positions[f"team_0_player_{i}"] = np.array([left_x, y], dtype=float)
            self.agent_positions[f"team_1_player_{i}"] = np.array([right_x, y], dtype=float)

        first_possessor = self.rng.choice([f"team_0_player_{i}" for i in range(self.num_players_per_team)])
        self.possession = first_possessor

        self.disc_position = self.agent_positions[self.possession].copy()
        self.possession_timer = 0
        self.throw_cooldown = {a: 0 for a in self.agents}

        self.disc_in_flight = False
        self.disc_target = None
        self.disc_direction = np.zeros(2)
        self.look_delay = int(self.rng.integers(2, 6))

        self._assign_defenders()
        self._pending_thrower = None
        self._prev_dist_to_goal = None

        self.prev_positions = {a: self.agent_positions[a].copy() for a in self.agents}

        return self._get_obs(), {a: {} for a in self.agents}

    # ---------------------------------------
    # Observations
    # ---------------------------------------
    def _get_obs(self):
        obs = {}
        norm_pos = {a: self._norm(self.agent_positions[a]) for a in self.agents}
        disc = self._norm(self.disc_position)

        for a in self.agents:
            team = self._team_of(a)
            teammates = [x for x in self.agents if self._team_of(x) == team and x != a]
            opponents = [x for x in self.agents if self._team_of(x) != team]

            o = [
                *norm_pos[a],
                *disc,
                float(self.possession == a),
                *(np.concatenate([norm_pos[x] for x in teammates]) if teammates else np.empty(0)),
                *(np.concatenate([norm_pos[x] for x in opponents]) if opponents else np.empty(0)),
            ]
            obs[a] = np.asarray(o, dtype=np.float32)

        return obs

    # ---------------------------------------
    # Step
    # ---------------------------------------
    def step(self, actions):
        self.steps += 1
        field_w, field_h = self.field_size

        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: self.steps >= self.max_steps for a in self.agents}
        infos = {a: {} for a in self.agents}

        prev_positions = {a: self.agent_positions[a].copy() for a in self.agents}

        for a in self.throw_cooldown:
            if self.throw_cooldown[a] > 0:
                self.throw_cooldown[a] -= 1

        if self.disc_in_flight:
            turnover = self._update_disc_flight(rewards)
            if turnover:
                terminations = {a: True for a in self.agents}
                return self._get_obs(), rewards, terminations, truncations, infos
        else:
            self.possession_timer += 1
            if self.possession_timer > self.possession_stall_limit:
                self._turnover_on_stall()
                terminations = {a: True for a in self.agents}
                return self._get_obs(), rewards, terminations, truncations, infos

        # Offense actions
        for agent in [a for a in self.agents if self._team_of(a) == 0]:
            action = actions.get(agent, 4 if agent == self.possession else 0)

            if agent == self.possession:
                if action == 4 and self.throw_cooldown[agent] == 0 and self.look_delay <= 0 and not self.disc_in_flight:
                    self._throw_disc(agent)
                else:
                    self.look_delay = max(0, self.look_delay - 1)
                continue

            dx = dy = 0
            if action == 0: dy = 2
            elif action == 1: dy = -2
            elif action == 2: dx = -2
            elif action == 3: dx = 2

            pos = self.agent_positions[agent]
            self.agent_positions[agent][0] = np.clip(pos[0] + dx, 0, field_w - 1)
            self.agent_positions[agent][1] = np.clip(pos[1] + dy, 0, field_h - 1)

        # Defenders move
        self._assign_defenders()
        self._move_defenders_keep_close(field_w, field_h)

        # PATCH B — Reward upfield movement for all offense players
        for a in [x for x in self.agents if self._team_of(x) == 0]:
            dx = self.agent_positions[a][0] - prev_positions[a][0]
            rewards[a] += 0.0003 * dx

        # PATCH C — Reward openness vs defender
        for d, o in self.defensive_marks.items():
            if o is None:
                continue
            dist = np.linalg.norm(self.agent_positions[o] - self.agent_positions[d])
            rewards[o] += 0.0002 * dist

        # PATCH D — Penalize stagnation
        for a in self.agents:
            if np.linalg.norm(self.agent_positions[a] - prev_positions[a]) < 0.01:
                rewards[a] -= 0.0001

        # Existing progress shaping
        if self.possession is not None and not self.disc_in_flight and self._team_of(self.possession) == 0:
            goal_x = self.endzone_depth + self.playing_length
            dist_to_goal = goal_x - self.agent_positions[self.possession][0]

            if self._prev_dist_to_goal is not None:
                progress = self._prev_dist_to_goal - dist_to_goal
                rewards[self.possession] += 0.001 * progress

            self._prev_dist_to_goal = dist_to_goal
        else:
            self._prev_dist_to_goal = None

        # Scoring
        if self.possession is not None:
            poss_team = self._team_of(self.possession)
            px = self.agent_positions[self.possession][0]
        else:
            poss_team = None

        if poss_team == 0 and px >= (self.endzone_depth + self.playing_length):
            rewards[self.possession] += 1.0
            terminations = {a: True for a in self.agents}

        if poss_team == 1 and px < self.endzone_depth:
            rewards[self.possession] += 1.0
            terminations = {a: True for a in self.agents}

        return self._get_obs(), rewards, terminations, truncations, infos

    # ---------------------------------------
    # Throw & Flight
    # ---------------------------------------
    def _throw_disc(self, agent):
        team = self._team_of(agent)
        teammates = [a for a in self.agents if self._team_of(a) == team and a != agent]
        if not teammates:
            return

        openness = []
        for t in teammates:
            tpos = self.agent_positions[t]
            nearest_def = min(
                np.linalg.norm(self.agent_positions[d] - tpos)
                for d in self.agents if self._team_of(d) != team
            )
            openness.append(nearest_def)

        probs = np.array(openness)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(teammates), 1 / len(teammates))

        target = self.rng.choice(teammates, p=probs)

        self.disc_in_flight = True
        self.disc_target = target
        self._pending_thrower = agent
        self.throw_cooldown[agent] = int(self.rng.integers(3, 7))
        self.look_delay = int(self.rng.integers(2, 6))

        direction = self.agent_positions[target] - self.agent_positions[agent]
        dist = np.linalg.norm(direction)

        # PATCH E — faster throw speed, always
        speed = 2.5 + self.rng.random()

        self.disc_direction = (direction / dist) * speed if dist > 1e-6 else np.zeros(2)

    # ---------------------------------------
    # Disc flight update
    # ---------------------------------------
    def _update_disc_flight(self, rewards):
        if not self.disc_in_flight:
            return False

        # Move disc
        self.disc_position += self.disc_direction
        self.disc_position[0] = np.clip(self.disc_position[0], 0, self.field_size[0] - 1)
        self.disc_position[1] = np.clip(self.disc_position[1], 0, self.field_size[1] - 1)

        # Catch
        if self.disc_target is not None:
            recv_pos = self.agent_positions[self.disc_target]
            if np.linalg.norm(recv_pos - self.disc_position) <= self.catch_range:
                if self._pending_thrower:
                    rewards[self._pending_thrower] += 0.05
                rewards[self.disc_target] += 0.05

                self.possession = self.disc_target
                self.disc_in_flight = False
                self.disc_target = None
                self.disc_direction = np.zeros(2)
                self.possession_timer = 0
                self._pending_thrower = None
                self._prev_dist_to_goal = None
                return False

        # Interception
        for d in [a for a in self.agents if self._team_of(a) == 1]:
            if np.linalg.norm(self.agent_positions[d] - self.disc_position) <= self.intercept_range:
                if self._pending_thrower:
                    rewards[self._pending_thrower] -= 0.1

                self.possession = d
                self.disc_in_flight = False
                self.disc_target = None
                self.disc_direction = np.zeros(2)
                self._pending_thrower = None
                self._prev_dist_to_goal = None
                return True

        # OOB turnover
        if (
            self.disc_position[0] <= 0 or
            self.disc_position[0] >= self.field_size[0] - 1 or
            self.disc_position[1] <= 0 or
            self.disc_position[1] >= self.field_size[1] - 1
        ):
            if self._pending_thrower:
                rewards[self._pending_thrower] -= 0.1

            defenders = [a for a in self.agents if self._team_of(a) == 1]
            if defenders:
                nearest_def = min(defenders, key=lambda a: np.linalg.norm(self.agent_positions[a] - self.disc_position))
                self.possession = nearest_def

            self.disc_in_flight = False
            self.disc_target = None
            self.disc_direction = np.zeros(2)
            self._pending_thrower = None
            self._prev_dist_to_goal = None
            return True

        return False

    # ---------------------------------------
    # Stall turnover
    # ---------------------------------------
    def _turnover_on_stall(self):
        if self.possession is None:
            return

        old_team = self._team_of(self.possession)
        defenders = [a for a in self.agents if self._team_of(a) != old_team]

        if defenders:
            nearest = min(defenders, key=lambda a: np.linalg.norm(self.agent_positions[a] - self.disc_position))
            self.possession = nearest

        self.disc_in_flight = False
        self.disc_target = None
        self.disc_direction = np.zeros(2)
        self.possession_timer = 0
        self._pending_thrower = None
        self._prev_dist_to_goal = None

    # ---------------------------------------
    # Defender logic
    # ---------------------------------------
    def _assign_defenders(self):
        defenders = [a for a in self.agents if self._team_of(a) == 1]
        offensives = [a for a in self.agents if self._team_of(a) == 0]

        if not defenders or not offensives:
            self.defensive_marks = {d: None for d in defenders}
            return

        if _HAS_SCIPY:
            D = np.array([self.agent_positions[d] for d in defenders])
            O = np.array([self.agent_positions[o] for o in offensives])
            cost = np.linalg.norm(D[:, None, :] - O[None, :, :], axis=-1)
            r, c = linear_sum_assignment(cost)
            self.defensive_marks = {defenders[i]: offensives[j] for i, j in zip(r, c)}
        else:
            offs_remaining = set(offensives)
            marks = {}
            for d in sorted(defenders, key=lambda dd: np.linalg.norm(self.agent_positions[dd] - self.disc_position)):
                if offs_remaining:
                    nearest = min(offs_remaining, key=lambda o: np.linalg.norm(self.agent_positions[d] - self.agent_positions[o]))
                    marks[d] = nearest
                    offs_remaining.remove(nearest)
                else:
                    nearest = min(offensives, key=lambda o: np.linalg.norm(self.agent_positions[d] - self.agent_positions[o]))
                    marks[d] = nearest

            self.defensive_marks = marks

    def _move_defenders_keep_close(self, field_w, field_h):
        for defender, mark in self.defensive_marks.items():
            if mark is None:
                continue

            dpos = self.agent_positions[defender]
            mpos = self.agent_positions[mark]
            disc = self.disc_position

            mid = (mpos + disc) / 2.0
            v = mid - mpos
            vnorm = np.linalg.norm(v)

            desired = mpos + (v / vnorm) * min(vnorm, self.defender_mark_distance) if vnorm > 1e-6 else mpos
            self.agent_positions[defender] = np.clip(desired, [0, 0], [field_w - 1, field_h - 1])

    # ---------------------------------------
    # Rendering
    # ---------------------------------------
    def render_matplotlib(self, block=False):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(12, 5))
            plt.ion()

        ax = self._ax
        ax.clear()

        field_w, field_h = self.field_size
        ax.set_xlim(-0.5, field_w - 0.5)
        ax.set_ylim(-0.5, field_h - 0.5)
        ax.set_aspect("equal")

        stall_text = f"Stall: {int(self.possession_timer)} / {self.possession_stall_limit}"
        ax.set_title(f"Ultimate Frisbee (Team 0 offense) — {stall_text}")

        ax.axvspan(0, self.endzone_depth, color="lightgreen", alpha=0.25)
        ax.axvspan(self.endzone_depth + self.playing_length, field_w, color="lightgreen", alpha=0.25)
        ax.axvline(self.endzone_depth, color="black", linestyle="--", alpha=0.7)
        ax.axvline(self.endzone_depth + self.playing_length, color="black", linestyle="--", alpha=0.7)

        for a, pos in self.agent_positions.items():
            team = self._team_of(a)
            color = "red" if team == 0 else "blue"
            is_poss = a == self.possession and not self.disc_in_flight
            ax.scatter(pos[0], pos[1], color=color, s=120 if is_poss else 70,
                       marker="*" if is_poss else "o", zorder=3)
            ax.text(pos[0] + 0.3, pos[1] + 0.3, a.split("_")[-1], fontsize=8)

        for d, mark in self.defensive_marks.items():
            if mark is not None:
                dpos, mpos = self.agent_positions[d], self.agent_positions[mark]
                ax.plot([dpos[0], mpos[0]], [dpos[1], mpos[1]],
                        linestyle="--", linewidth=1.0, alpha=0.5, color="gray")

        if self.disc_in_flight and self.disc_target is not None:
            target_pos = self.agent_positions[self.disc_target]
            ax.plot([self.disc_position[0], target_pos[0]],
                    [self.disc_position[1], target_pos[1]],
                    linestyle=":", color="orange", alpha=0.6, zorder=4)
            ax.scatter(self.disc_position[0], self.disc_position[1],
                       color="yellow", s=140, marker="o", zorder=5)
        elif self.possession:
            dp = self.agent_positions[self.possession]
            ax.scatter(dp[0], dp[1], s=220, facecolors='none',
                       edgecolors='orange', linewidths=2, zorder=4)

        plt.draw()
        plt.pause(0.05)
        if block:
            plt.ioff()
            plt.show()

    # ---------------------------------------
    # Close
    # ---------------------------------------
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
