from envs.dynamic_ultimate_frisbee import DynamicUltimateFrisbeeEnv

env = DynamicUltimateFrisbeeEnv(use_regulation_field=True)
obs, info = env.reset()

obs, info = env.reset()
done = False
while not done:
    actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
    obs, rewards, terminations, truncations, info = env.step(actions)
    env.render_matplotlib()
    done = all(terminations.values()) or all(truncations.values())

# Show final frame and block
env.render_matplotlib(block=True)

