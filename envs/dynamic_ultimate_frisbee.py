from pettingzoo import ParallelEnv
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


REWARD_SCORE = 1.0
REWARD_CATCH = 0.05
REWARD_PASSER_CATCH = 0.05
REWARD_INTERCEPT = -0.1
REWARD_STALL = -1.0
REWARD_PROGRESS = 0.001
REWARD_UPFIELD = 0.0003
REWARD_OPENNESS = 0.0002
REWARD_STAGNATION = -0.0001

THROW_THRESHOLD = 0.5
THROW_SPEED_MIN = 2.5
THROW_SPEED_VAR = 1.0

MOVE_SPEED = 1.0


class DynamicUltimateFrisbeeEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "ultimate_frisbee_v3"}

    def __init__(
        self,
        num_players_per_team=2,
        use_regulation_field=False,
        seed=None,
        render_mode=None,
        debug=False,
        max_steps=400,
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

        if use_regulation_field:
            self.endzone_depth = 20
            self.playing_length = 70
            self.playing_width = 40
            self.field_size = (self.playing_length + 2 * self.endzone_depth, self.playing_width)
        else:
            self.endzone_depth = 3
            self.playing_length = 24
            self.playing_width = 20
            self.field_size = (self.playing_length + 2 * self.endzone_depth, self.playing_width)

        self.agents = [f"team_{t}_player_{i}" for t in range(2) for i in range(num_players_per_team)]

        obs_dim = self._obs_dim()
        self.observation_spaces = {a: spaces.Box(0, 1, (obs_dim,), np.float32) for a in self.agents}
        self.action_spaces = {a: spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32) for a in self.agents}

        self.intercept_range = 1.2
        self.catch_range = 4.0
        self.defender_mark_distance = 2.0
        self.possession_stall_limit = 12

        self._fig = None
        self._ax = None

        self.reset()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _team_of(self, agent):
        return int(agent.split("_")[1])

    def _obs_dim(self):
        t = self.num_players_per_team
        return 2 + 2 + 1 + 2 * (t - 1) + 2 * t + 1

    def _norm(self, xy):
        w, h = self.field_size
        return np.array([xy[0] / (w - 1), xy[1] / (h - 1)], np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed

        self.steps = 0
        fw, fh = self.field_size

        ys = np.linspace(0, fh - 1, self.num_players_per_team)
        self.agent_positions = {}

        left_x = self.endzone_depth
        right_x = self.endzone_depth + self.playing_length

        for i, y in enumerate(ys):
            self.agent_positions[f"team_0_player_{i}"] = np.array([left_x, y], float)
            self.agent_positions[f"team_1_player_{i}"] = np.array([right_x, y], float)

        first = self.rng.choice([f"team_0_player_{i}" for i in range(self.num_players_per_team)])
        self.possession = first

        self.disc_position = self.agent_positions[first].copy()
        self.disc_velocity = np.zeros(2)
        self.disc_target = None
        self.disc_in_flight = False

        self.possession_timer = 0

        self._pending_thrower = None
        self._prev_dist_to_goal = None

        self._assign_defenders()

        return self._get_obs(), {a: {} for a in self.agents}

    def _get_obs(self):
        obs = {}
        norm_pos = {a: self._norm(self.agent_positions[a]) for a in self.agents}
        disc = self._norm(self.disc_position)
        stall_norm = self.possession_timer / self.possession_stall_limit

        for a in self.agents:
            team = self._team_of(a)
            mates = [x for x in self.agents if self._team_of(x) == team and x != a]
            opps = [x for x in self.agents if self._team_of(x) != team]

            vec = [
                *norm_pos[a],
                *disc,
                float(self.possession == a),
                *(np.concatenate([norm_pos[m] for m in mates]) if mates else []),
                *(np.concatenate([norm_pos[o] for o in opps]) if opps else []),
                stall_norm,
            ]
            obs[a] = np.asarray(vec, np.float32)

        return obs

    def step(self, actions):
        self.steps += 1
        fw, fh = self.field_size

        rewards = {a: 0.0 for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: self.steps >= self.max_steps for a in self.agents}
        infos = {a: {} for a in self.agents}

        prev = {a: self.agent_positions[a].copy() for a in self.agents}

        if self.debug:
            print(f"\nSTEP {self.steps} | possession={self.possession} | stall={self.possession_timer}")

        if self.disc_in_flight:
            turnover = self._update_disc_flight(rewards)
            if turnover:
                term = {a: True for a in self.agents}
                return self._get_obs(), rewards, term, trunc, infos
        else:
            self.possession_timer += 1
            if self.possession_timer > self.possession_stall_limit:
                for a in self.agents:
                    if self._team_of(a) == 0:
                        rewards[a] += REWARD_STALL
                if self.debug:
                    print(f"[STALL] offense stalled at step {self.steps}")
                term = {a: True for a in self.agents}
                return self._get_obs(), rewards, term, trunc, infos

        for a in self.agents:
            team = self._team_of(a)
            ax, ay, throw_raw = actions.get(a, np.zeros(3, dtype=np.float32))

            if a != self.possession:
                dx = float(ax) * MOVE_SPEED
                dy = float(ay) * MOVE_SPEED
                pos = self.agent_positions[a]
                pos[0] = np.clip(pos[0] + dx, 0, fw - 1)
                pos[1] = np.clip(pos[1] + dy, 0, fh - 1)
            else:
                if throw_raw > THROW_THRESHOLD and not self.disc_in_flight:
                    self._throw_disc(a, actions)
                else:
                    dx = float(ax) * 0.2
                    dy = float(ay) * 0.2
                    pos = self.agent_positions[a]
                    pos[0] = np.clip(pos[0] + dx, 0, fw - 1)
                    pos[1] = np.clip(pos[1] + dy, 0, fh - 1)

        self._assign_defenders()
        self._move_defenders(fw, fh)

        for a in self.agents:
            if self._team_of(a) == 0:
                dx = self.agent_positions[a][0] - prev[a][0]
                rewards[a] += REWARD_UPFIELD * dx

        for d, o in self.defensive_marks.items():
            if o is not None:
                dist = np.linalg.norm(self.agent_positions[o] - self.agent_positions[d])
                rewards[o] += REWARD_OPENNESS * dist

        for a in self.agents:
            if np.linalg.norm(self.agent_positions[a] - prev[a]) < 0.01:
                rewards[a] += REWARD_STAGNATION

        if self.possession and not self.disc_in_flight:
            px = self.agent_positions[self.possession][0]
            goal_x = self.endzone_depth + self.playing_length
            dist = goal_x - px
            if self._prev_dist_to_goal is not None:
                prog = self._prev_dist_to_goal - dist
                rewards[self.possession] += REWARD_PROGRESS * prog
            self._prev_dist_to_goal = dist

            if px >= goal_x:
                rewards[self.possession] += REWARD_SCORE
                term = {a: True for a in self.agents}
                if self.debug:
                    print(f"[SCORE] {self.possession} scored")

        return self._get_obs(), rewards, term, trunc, infos

    def _throw_disc(self, agent, actions):
        team = self._team_of(agent)
        mates = [a for a in self.agents if self._team_of(a) == team and a != agent]
        if not mates:
            return

        logits = []
        mate_positions = []
        for m in mates:
            mx, my, _ = actions.get(m, np.zeros(3, dtype=np.float32))
            tgt_pos = self.agent_positions[m] + np.array([mx, my], float)
            mate_positions.append(tgt_pos)
            nearest_def = min(
                np.linalg.norm(self.agent_positions[d] - tgt_pos)
                for d in self.agents if self._team_of(d) != team
            )
            logits.append(nearest_def)

        logits = np.asarray(logits, float)
        probs = logits / logits.sum() if logits.sum() > 0 else np.full(len(mates), 1 / len(mates))

        idx = self.rng.choice(len(mates), p=probs)
        target = mates[idx]
        target_pos = mate_positions[idx]

        if self.debug:
            print(f"[THROW] {agent} -> {target}")

        self.disc_in_flight = True
        self.disc_target = target
        self._pending_thrower = agent
        self.possession = None

        direction = target_pos - self.agent_positions[agent]
        dist = np.linalg.norm(direction)
        speed = THROW_SPEED_MIN + self.rng.random() * THROW_SPEED_VAR
        self.disc_velocity = (direction / dist) * speed if dist > 1e-6 else np.zeros(2)

    def _update_disc_flight(self, rewards):
        self.disc_position += self.disc_velocity
        fw, fh = self.field_size
        x, y = self.disc_position

        if x <= 0 or x >= fw - 1 or y <= 0 or y >= fh - 1:
            if self._pending_thrower:
                rewards[self._pending_thrower] += REWARD_INTERCEPT
            defenders = [a for a in self.agents if self._team_of(a) == 1]
            if defenders:
                nearest = min(defenders, key=lambda d: np.linalg.norm(self.agent_positions[d] - self.disc_position))
                self.possession = nearest
            self.disc_in_flight = False
            self.disc_velocity = np.zeros(2)
            self.disc_target = None
            self._pending_thrower = None
            self._prev_dist_to_goal = None
            if self.debug:
                print("[TURNOVER] out of bounds")
            return True

        if self.disc_target is not None:
            recv = self.agent_positions[self.disc_target]
            if np.linalg.norm(recv - self.disc_position) <= self.catch_range:
                if self._pending_thrower:
                    rewards[self._pending_thrower] += REWARD_PASSER_CATCH
                rewards[self.disc_target] += REWARD_CATCH
                self.possession = self.disc_target
                self.disc_in_flight = False
                self.disc_velocity = np.zeros(2)
                self.disc_target = None
                self._pending_thrower = None
                self.possession_timer = 0
                self._prev_dist_to_goal = None
                if self.debug:
                    print(f"[CATCH] by {self.possession}")
                return False

        for d in [a for a in self.agents if self._team_of(a) == 1]:
            if np.linalg.norm(self.agent_positions[d] - self.disc_position) <= self.intercept_range:
                if self._pending_thrower:
                    rewards[self._pending_thrower] += REWARD_INTERCEPT
                self.possession = d
                self.disc_in_flight = False
                self.disc_velocity = np.zeros(2)
                self.disc_target = None
                self._pending_thrower = None
                self._prev_dist_to_goal = None
                if self.debug:
                    print(f"[INTERCEPT] by {d}")
                return True

        return False

    def _assign_defenders(self):
        defenders = [a for a in self.agents if self._team_of(a) == 1]
        offs = [a for a in self.agents if self._team_of(a) == 0]
        if not defenders or not offs:
            self.defensive_marks = {d: None for d in defenders}
            return

        if _HAS_SCIPY:
            D = np.array([self.agent_positions[d] for d in defenders])
            O = np.array([self.agent_positions[o] for o in offs])
            cost = np.linalg.norm(D[:, None, :] - O[None, :, :], axis=-1)
            r, c = linear_sum_assignment(cost)
            self.defensive_marks = {defenders[i]: offs[j] for i, j in zip(r, c)}
        else:
            rem = set(offs)
            marks = {}
            for d in defenders:
                if rem:
                    nearest = min(rem, key=lambda o: np.linalg.norm(self.agent_positions[d] - self.agent_positions[o]))
                    rem.remove(nearest)
                else:
                    nearest = min(offs, key=lambda o: np.linalg.norm(self.agent_positions[d] - self.agent_positions[o]))
                marks[d] = nearest
            self.defensive_marks = marks

    def _move_defenders(self, fw, fh):
        for d, o in self.defensive_marks.items():
            if o is None:
                continue
            dpos = self.agent_positions[d]
            opos = self.agent_positions[o]
            disc = self.disc_position
            mid = (opos + disc) / 2.0
            v = mid - opos
            n = np.linalg.norm(v)
            desired = opos + (v / n) * min(n, self.defender_mark_distance) if n > 1e-6 else opos
            dpos[:] = np.clip(desired, [0, 0], [fw - 1, fh - 1])

    def render_matplotlib(self, block=False):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(12, 6))
            plt.ion()
        ax = self._ax
        ax.clear()

        fw, fh = self.field_size
        ax.set_xlim(-0.5, fw - 0.5)
        ax.set_ylim(-0.5, fh - 0.5)
        ax.set_aspect("equal")

        stall_text = f"Stall: {int(self.possession_timer)} / {self.possession_stall_limit}"
        ax.set_title(f"Ultimate Frisbee (Team 0 offense) — {stall_text}")

        ax.axvspan(0, self.endzone_depth, color="lightgreen", alpha=0.25)
        ax.axvspan(self.endzone_depth + self.playing_length, fw, color="lightgreen", alpha=0.25)
        ax.axvline(self.endzone_depth, color="black", linestyle="--", alpha=0.7)
        ax.axvline(self.endzone_depth + self.playing_length, color="black", linestyle="--", alpha=0.7)

        for a, pos in self.agent_positions.items():
            team = self._team_of(a)
            color = "red" if team == 0 else "blue"
            is_poss = (a == self.possession and not self.disc_in_flight)
            ax.scatter(pos[0], pos[1], color=color, s=120 if is_poss else 70,
                       marker="*" if is_poss else "o", zorder=3)
            ax.text(pos[0] + 0.3, pos[1] + 0.3, a.split("_")[-1], fontsize=8)

        for d, mark in self.defensive_marks.items():
            if mark is not None:
                dpos = self.agent_positions[d]
                mpos = self.agent_positions[mark]
                ax.plot([dpos[0], mpos[0]], [dpos[1], mpos[1]],
                        linestyle="--", linewidth=1.0, alpha=0.5, color="gray")

        if self.disc_in_flight:
            ax.scatter(self.disc_position[0], self.disc_position[1],
                       color="yellow", s=140, marker="o", zorder=5)
            if self.disc_target is not None:
                tpos = self.agent_positions[self.disc_target]
                ax.plot([self.disc_position[0], tpos[0]],
                        [self.disc_position[1], tpos[1]],
                        linestyle=":", color="orange", alpha=0.6, zorder=4)
        elif self.possession:
            dp = self.agent_positions[self.possession]
            ax.scatter(dp[0], dp[1], s=220, facecolors='none',
                       edgecolors='orange', linewidths=2, zorder=4)
            ax.scatter(dp[0], dp[1], color="yellow", s=80, marker="o", zorder=5)

        plt.draw()
        plt.pause(0.03)
        if block:
            plt.ioff()
            plt.show()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
