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
REWARD_CATCH = 0.15
REWARD_PASSER_CATCH = 0.15
REWARD_INTERCEPT = -0.2
REWARD_STALL = -1.0
REWARD_PROGRESS = 0.01
REWARD_UPFIELD = 0.03
REWARD_OPENNESS = 0.0002
REWARD_STAGNATION = -0.0001
REWARD_THROW_BACKWARD = -0.02

THROW_THRESHOLD = 0.2
THROW_SPEED_MIN = 3.0
THROW_SPEED_VAR = 1.2

PLAYER_MAX_SPEED = 0.8
PLAYER_ACCEL     = 0.35
PLAYER_FRICTION  = 0.88


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
        else:
            self.endzone_depth = 3
            self.playing_length = 24
            self.playing_width = 20

        self.field_size = (self.playing_length + 2 * self.endzone_depth, self.playing_width)

        self.agents = [f"team_{t}_player_{i}" for t in range(2) for i in range(num_players_per_team)]

        obs_dim = self._obs_dim()
        self.observation_spaces = {a: spaces.Box(0, 1, (obs_dim,), np.float32) for a in self.agents}
        self.action_spaces = {a: spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32) for a in self.agents}

        self.intercept_range = 1.0
        self.catch_range = 2
        self.defender_mark_distance = 2.0
        self.possession_stall_limit = 30

        self.throw_windup = 0
        self.throw_pending_target = None
        self.min_throw_distance = 5.0

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
        quarter_x = self.endzone_depth + self.playing_length * 0.25

        for i, y in enumerate(ys):
            self.agent_positions[f"team_0_player_{i}"] = np.array([left_x, y], float)
            self.agent_positions[f"team_1_player_{i}"] = np.array([quarter_x, y], float)

        self.player_vel = {a: np.zeros(2) for a in self.agents}

        first = self.rng.choice([f"team_0_player_{i}" for i in range(self.num_players_per_team)])
        self.possession = first

        self.disc_position = self.agent_positions[first].copy()
        self.disc_velocity = np.zeros(2)
        self.disc_target = None
        self.disc_in_flight = False

        self.throw_windup = 0
        self.throw_pending_target = None

        self.possession_timer = 0
        self._pending_thrower = None
        self._pending_backward_penalty = 0.0
        self._prev_dist_to_goal = None

        self._assign_defenders()
        self._last_opos = {}

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

        if self.throw_windup > 0:
            self.throw_windup -= 1
            if self.throw_windup == 0 and self.throw_pending_target is not None:
                self._execute_throw()

        if self._pending_thrower is not None:
            rewards[self._pending_thrower] += self._pending_backward_penalty
            self._pending_backward_penalty = 0.0

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
                for a in self.agents:
                    term[a] = True
                return self._get_obs(), rewards, term, trunc, infos

        for a in self.agents:
            ax, ay, throw_raw = actions.get(a, np.zeros(3))

            if a != self.possession:
                if self.disc_in_flight and a == self.disc_target:
                    future_disc = self.disc_position + self.disc_velocity
                    chase_vec = future_disc - self.agent_positions[a]
                    dist = np.linalg.norm(chase_vec)
                    chase_dir = chase_vec / dist if dist > 1e-6 else np.zeros(2)
                    v = chase_dir * PLAYER_MAX_SPEED
                    self.player_vel[a] = v
                    pos = self.agent_positions[a]
                    pos += v
                    pos[:] = np.clip(pos, [0, 0], [fw-1, fh-1])
                    continue
                accel = np.array([ax, ay]) * PLAYER_ACCEL
                v = self.player_vel[a] + accel
                speed = np.linalg.norm(v)
                if speed > PLAYER_MAX_SPEED:
                    v = (v / speed) * PLAYER_MAX_SPEED
                v *= PLAYER_FRICTION
                self.player_vel[a] = v
                pos = self.agent_positions[a]
                pos += v
                pos[:] = np.clip(pos, [0,0], [fw-1,fh-1])

            else:
                self.player_vel[a] = np.zeros(2)
                rewards[a] -= 0.02

                if throw_raw > THROW_THRESHOLD and not self.disc_in_flight and self.throw_windup == 0:
                    chosen = self._select_throw_target(a, actions)
                    if chosen is not None:
                        self.throw_pending_target = chosen
                        self.throw_windup = 2

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

        offense = [a for a in self.agents if self._team_of(a)==0]
        for i,a in enumerate(offense):
            pos_a = self.agent_positions[a]
            for b in offense[i+1:]:
                pos_b = self.agent_positions[b]
                diff = pos_a - pos_b
                dist = np.linalg.norm(diff)
                if dist < 4.0 and dist > 1e-6:
                    repulse = (diff/dist)*0.10
                    self.agent_positions[a] += repulse
                    self.agent_positions[b] -= repulse
                    self.agent_positions[a][:] = np.clip(self.agent_positions[a],[0,0],[fw-1,fh-1])
                    self.agent_positions[b][:] = np.clip(self.agent_positions[b],[0,0],[fw-1,fh-1])

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
                for a in self.agents:
                    term[a] = True

        return self._get_obs(), rewards, term, trunc, infos

    def _select_throw_target(self, agent, actions):
        team = self._team_of(agent)
        mates = [a for a in self.agents if self._team_of(a)==team and a!=agent]

        logits = []
        pred_positions = []
        thrower_pos = self.agent_positions[agent]

        for m in mates:
            raw_dist = np.linalg.norm(self.agent_positions[m] - thrower_pos)
            if raw_dist < self.min_throw_distance:
                continue

            mx, my, _ = actions.get(m, np.zeros(3))
            accel = np.array([mx, my]) * PLAYER_ACCEL
            v = self.player_vel[m] + accel
            s = np.linalg.norm(v)
            if s > PLAYER_MAX_SPEED:
                v = (v/s)*PLAYER_MAX_SPEED

            disc_speed = THROW_SPEED_MIN + self.rng.random()*THROW_SPEED_VAR
            t = raw_dist/disc_speed
            predicted = self.agent_positions[m] + v*t
            pred_positions.append((m, predicted))

            nearest_def = min(
                np.linalg.norm(self.agent_positions[d] - predicted)
                for d in self.agents if self._team_of(d)!=team
            )
            progress = max(0.0, predicted[0] - thrower_pos[0])
            logits.append((nearest_def + 0.2*progress, m))

        if not logits:
            return None

        arr = np.array([l[0] for l in logits])
        probs = arr/arr.sum() if arr.sum()>0 else np.ones(len(arr))/len(arr)
        idx = self.rng.choice(len(arr), p=probs)
        return logits[idx][1]

    def _execute_throw(self):
        agent = self.possession
        if agent is None or self.throw_pending_target is None:
            return

        target = self.throw_pending_target
        self.throw_pending_target = None

        thrower_pos = self.agent_positions[agent]
        target_pos = self.agent_positions[target]
        raw_dist = np.linalg.norm(target_pos - thrower_pos)

        thrower_x = thrower_pos[0]
        if target_pos[0] < thrower_x:
            self._pending_backward_penalty = REWARD_THROW_BACKWARD * (thrower_x - target_pos[0])
        else:
            self._pending_backward_penalty = 0.0

        speed = THROW_SPEED_MIN + self.rng.random()*THROW_SPEED_VAR
        desired_vel = target_pos - thrower_pos
        d = np.linalg.norm(desired_vel)
        v = desired_vel/d * speed if d>1e-6 else np.zeros(2)

        self.disc_velocity = 0.6*self.disc_velocity + 0.4*v
        self.disc_in_flight = True
        self._pending_thrower = agent
        self.disc_target = target
        self.possession = None

    def _update_disc_flight(self, rewards):
        self.disc_position += self.disc_velocity
        fw, fh = self.field_size
        x,y = self.disc_position

        if x<=0 or x>=fw-1 or y<=0 or y>=fh-1:
            if self._pending_thrower:
                rewards[self._pending_thrower] += REWARD_INTERCEPT
            defenders = [a for a in self.agents if self._team_of(a)==1]
            self.possession = min(defenders, key=lambda d: np.linalg.norm(self.agent_positions[d]-self.disc_position))
            self.disc_in_flight=False
            self.disc_velocity=np.zeros(2)
            self.disc_target=None
            self._pending_thrower=None
            self._prev_dist_to_goal=None
            return True

        if self.disc_target is not None:
            recv = self.agent_positions[self.disc_target]
            if np.linalg.norm(recv-self.disc_position) <= self.catch_range:
                thrower = self._pending_thrower
                gain = recv[0] - self.agent_positions[thrower][0]
                if gain<0:
                    rewards[thrower]+=REWARD_THROW_BACKWARD*abs(gain)
                rewards[self.disc_target]+=REWARD_CATCH
                rewards[thrower]+=REWARD_PASSER_CATCH
                if gain>0:
                    rewards[thrower]+=0.03*gain
                self.possession=self.disc_target
                self.disc_in_flight=False
                self.disc_velocity=np.zeros(2)
                self.disc_target=None
                self._pending_thrower=None
                self.possession_timer=0
                self._prev_dist_to_goal=None
                return False

        for d in [a for a in self.agents if self._team_of(a)==1]:
            if np.linalg.norm(self.agent_positions[d]-self.disc_position)<=self.intercept_range:
                if self._pending_thrower:
                    rewards[self._pending_thrower]+=REWARD_INTERCEPT
                self.possession=d
                self.disc_in_flight=False
                self.disc_velocity=np.zeros(2)
                self.disc_target=None
                self._pending_thrower=None
                self._prev_dist_to_goal=None
                return True

        return False

    def _assign_defenders(self):
        if not hasattr(self, "persistent_marks"):
            defenders = [a for a in self.agents if self._team_of(a)==1]
            offs = [a for a in self.agents if self._team_of(a)==0]

            if _HAS_SCIPY:
                D = np.array([self.agent_positions[d] for d in defenders])
                O = np.array([self.agent_positions[o] for o in offs])
                cost = np.linalg.norm(D[:,None,:] - O[None,:,:], axis=-1)
                r,c = linear_sum_assignment(cost)
                self.persistent_marks = {defenders[i]:offs[j] for i,j in zip(r,c)}
            else:
                self.persistent_marks = {d:offs[i%len(offs)] for i,d in enumerate(defenders)}

        self.defensive_marks = self.persistent_marks.copy()

    def _move_defenders(self, fw, fh):
        reaction_speed=0.5
        chase_speed=PLAYER_MAX_SPEED*0.9
        anticipation=0.3

        if not hasattr(self,"_def_vel"):
            self._def_vel={d:np.zeros(2) for d in self.defensive_marks}

        for d,o in self.defensive_marks.items():
            if o is None:
                continue

            dpos=self.agent_positions[d]
            opos=self.agent_positions[o]

            disc_x,disc_y=self.disc_position
            disc_to_cutter=opos-self.disc_position
            dist=np.linalg.norm(disc_to_cutter)
            if dist>1e-6:
                lane_block=self.disc_position+(disc_to_cutter/dist)*(0.6*dist)
            else:
                lane_block=opos.copy()

            desired_pos=lane_block.copy()
            desired_pos[1]=opos[1]+0.3*(opos[1]-disc_y)

            if o in self._last_opos:
                cutter_vel = opos-self._last_opos[o]
                desired_pos += anticipation*cutter_vel

            move_vec = desired_pos-dpos
            n=np.linalg.norm(move_vec)
            if n>1e-6:
                move_vec/=n

            self._def_vel[d]=reaction_speed*move_vec*chase_speed+(1-reaction_speed)*self._def_vel[d]
            new_pos=dpos+self._def_vel[d]
            self.agent_positions[d]=np.clip(new_pos,[0,0],[fw-1,fh-1])

        self._last_opos={o:self.agent_positions[o].copy() for o in self.defensive_marks.values() if o is not None}

    def render_matplotlib(self, block=False):
        if self._fig is None:
            self._fig,self._ax=plt.subplots(figsize=(12,6))
            plt.ion()
        ax=self._ax
        ax.clear()

        fw,fh=self.field_size
        ax.set_xlim(-0.5,fw-0.5)
        ax.set_ylim(-0.5,fh-0.5)
        ax.set_aspect("equal")

        stall_text=f"Stall: {int(self.possession_timer)} / {self.possession_stall_limit}"
        ax.set_title(f"Ultimate Frisbee (Team 0 offense) — {stall_text}")

        ax.axvspan(0,self.endzone_depth,color="lightgreen",alpha=0.25)
        ax.axvspan(self.endzone_depth+self.playing_length,fw,color="lightgreen",alpha=0.25)
        ax.axvline(self.endzone_depth,color="black",linestyle="--",alpha=0.7)
        ax.axvline(self.endzone_depth+self.playing_length,color="black",linestyle="--",alpha=0.7)

        for a,pos in self.agent_positions.items():
            team=self._team_of(a)
            color="red" if team==0 else "blue"
            is_poss=(a==self.possession and not self.disc_in_flight)
            ax.scatter(pos[0],pos[1],color=color,s=120 if is_poss else 70,
                       marker="*" if is_poss else "o",zorder=3)
            ax.text(pos[0]+0.3,pos[1]+0.3,a.split("_")[-1],fontsize=8)

        for d,mark in self.defensive_marks.items():
            if mark is not None:
                dpos=self.agent_positions[d]
                mpos=self.agent_positions[mark]
                ax.plot([dpos[0],mpos[0]],[dpos[1],mpos[1]],
                        linestyle="--",linewidth=1.0,alpha=0.5,color="gray")

        if self.disc_in_flight:
            ax.scatter(self.disc_position[0],self.disc_position[1],
                       color="yellow",s=140,marker="o",zorder=5)
            if self.disc_target is not None:
                tpos=self.agent_positions[self.disc_target]
                ax.plot([self.disc_position[0],tpos[0]],
                        [self.disc_position[1],tpos[1]],
                        linestyle=":",color="orange",alpha=0.6,zorder=4)
        elif self.possession:
            dp=self.agent_positions[self.possession]
            ax.scatter(dp[0],dp[1],s=220,facecolors='none',
                       edgecolors='orange',linewidths=2,zorder=4)
            ax.scatter(dp[0],dp[1],color="yellow",s=80,marker="o",zorder=5)

        plt.draw()
        plt.pause(0.03)
        if block:
            plt.ioff()
            plt.show()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig=None
            self._ax=None
