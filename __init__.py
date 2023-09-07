from gym.envs.registration import register

# register(
#     id = 'Tetris-v0',
#     entry_point = 'Tetris.envs:tetris',
#     max_episode_steps=300,
# )

register(
	id='Tetris-v0',
    entry_point='Tetris.envs.tetris:TetrisEnv',
)