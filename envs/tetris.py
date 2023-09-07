import gym
from gym import spaces
import numpy as np
from random import randrange as rand
import pygame, sys

config = {
	'cell_size':	20,
	'cols':		10,
	'rows':		20,
	'delay':	750,
	'maxfps':	30
}

colors = [
(0,   0,   0  ),
(255, 0,   0  ),
(0,   150, 0  ),
(0,   0,   255),
(255, 120, 0  ),
(255, 255, 0  ),
(180, 0,   255),
(0,   220, 220)
]

# Define the shapes of the single parts
tetris_shapes = [
	[[1, 1, 1],
	 [0, 1, 0]],
	
	[[0, 2, 2],
	 [2, 2, 0]],
	
	[[3, 3, 0],
	 [0, 3, 3]],
	
	[[4, 0, 0],
	 [4, 4, 4]],
	
	[[0, 0, 5],
	 [5, 5, 5]],
	
	[[6, 6, 6, 6]],
	
	[[7, 7],
	 [7, 7]]
]

# def register():
# 	gym.envs.register(
# 		id='Tetris-v0',
# 		entry_point='Tetris.envs:tetris:TetrisEnv'
# 	)

def rotate_clockwise(shape):
	return [ [ shape[y][x]
			for y in range(len(shape)) ]
		for x in range(len(shape[0]) - 1, -1, -1) ]

def check_collision(board, shape, offset):
	off_x, off_y = offset
	for cy, row in enumerate(shape):
		for cx, cell in enumerate(row):
			try:
				if cell and board[ cy + off_y ][ cx + off_x ]:
					return True
			except IndexError:
				return True
	return False

def remove_row(board, row):
	del board[row]
	return [[0 for i in range(config['cols'])]] + board
	
def join_matrixes(mat1, mat2, mat2_off):
	off_x, off_y = mat2_off
	for cy, row in enumerate(mat2):
		for cx, val in enumerate(row):
			mat1[cy+off_y-1	][cx+off_x] += val
	return mat1

def new_board():
	board = [ [ 0 for x in range(config['cols']) ]
			for y in range(config['rows']) ]
	board += [[ 1 for x in range(config['cols'])]]
	
	return board

def get_board_heights(board):
	boardHeights = []

	for i in range(10):
		colHeight = 0
		for j in range(20):
			if board[j][i] != 0:
				colHeight += 1
		boardHeights.append(colHeight)

	return boardHeights

class TetrisEnv(gym.Env):
	def __init__(self, state_dim=40, action_dim=(10,20)):
		pygame.init()
		# pygame.key.set_repeat(250,25)

		self.gameover = False
		self.paused = False
		self.blocks_placed = 0
		self.blocks_placed_limit = 100
		
		self.width = config['cell_size']*config['cols']
		self.height = config['cell_size']*config['rows']
		self.next_stones = []

		self.board = new_board()
		self.new_stone()
		
		# self.screen = pygame.display.set_mode((self.width, self.height))
		
		self.action_space = spaces.Discrete(state_dim)
		self.observation_space = spaces.Box(low=0, high=7, shape=(10,20), dtype=np.int32)
		
		pygame.event.set_blocked(pygame.MOUSEMOTION)

		# self.init_game()

	def reset(self):
		self.init_game()

		return np.array(self.board)[:20, :]

	def step(self, action):
		x_dest = action % 10
		r_dest = int(action / 10)

		self.blocks_placed += 1

		results = self.move_to_placement([x_dest, r_dest])
		lines_cleared = results[0]
		result_board = results[1]

		return [result_board, self.stone], lines_cleared, self.gameover, {}

	def render(self):
		pass

	def new_stone(self):
		self.next_stones.append(rand(len(tetris_shapes)))

		self.stone = tetris_shapes[self.next_stones.pop(0)]
		self.stone_x = int(config['cols'] / 2 - len(self.stone[0])/2)
		self.stone_y = 0
		
		if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
			self.gameover = True
	
	def init_game(self):
		self.gameover = False
		self.paused = False
		self.blocks_placed = 0
		self.blocks_placed_limit = 100

		self.next_stones = [rand(len(tetris_shapes))]
		self.next_stones.append(rand(len(tetris_shapes)))
		self.next_stones.append(rand(len(tetris_shapes)))

		self.board = new_board()
		self.new_stone()
	
	def center_msg(self, msg):
		for i, line in enumerate(msg.splitlines()):
			msg_image =  pygame.font.Font(
				pygame.font.get_default_font(), 12).render(
					line, False, (255,255,255), (0,0,0))
		
			msgim_center_x, msgim_center_y = msg_image.get_size()
			msgim_center_x //= 2
			msgim_center_y //= 2
		
			# self.screen.blit(msg_image, (
			#   self.width // 2-msgim_center_x,
			#   self.height // 2-msgim_center_y+i*22))
	
	# def draw_matrix(self, matrix, offset):
	# 	off_x, off_y  = offset
	# 	for y, row in enumerate(matrix):
	# 		for x, val in enumerate(row):
	# 			if val:
	# 				pygame.draw.rect(self.screen,
	# 					colors[val],
	# 					pygame.Rect(
	# 						(off_x+x) *
	# 						  config['cell_size'],
	# 						(off_y+y) *
	# 						  config['cell_size'], 
	# 						config['cell_size'],
	# 						config['cell_size']),0)
	
	def move_to_placement(self, movement):
		x_dest = movement[0]
		r_dest = movement[1]
		delta_x = x_dest - self.stone_x
		# self.toggle_pause()

		self.move(delta_x)
		
		for _ in range(r_dest):
			self.stone = rotate_clockwise(self.stone)
		
		lines_cleared = self.hardDrop()
		result_board = np.array(self.board)[:20, :]

		# self.toggle_pause()

		return [lines_cleared, result_board]
	
	def move(self, delta_x):
		new_x = self.stone_x + delta_x
		if new_x < 0:
			new_x = 0
		if new_x > config['cols'] - len(self.stone[0]):
			new_x = config['cols'] - len(self.stone[0])
		if not check_collision(self.board, self.stone, (new_x, self.stone_y)):
			self.stone_x = new_x

	def quit(self):
		self.center_msg("Exiting...")
		pygame.display.update()
		sys.exit()
	
	def drop(self):
		self.stone_y += 1
		if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
			self.board = join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
			self.new_stone()
			while True:
				for i, row in enumerate(self.board[:-1]):
					if 0 not in row:
						self.board = remove_row(self.board, i)
						break
				else:
					break
	
	def hardDrop(self):
		lines_cleared = 0

		# if not self.gameover and not self.paused:
		while not check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
			self.stone_y += 1
		self.board = join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
		self.new_stone()
		while True:
			for i, row in enumerate(self.board[:-1]):
				if 0 not in row:
					self.board = remove_row(self.board, i)
					lines_cleared += 1
					break
			else:
				break

		return lines_cleared
	
	def rotate_stone(self):
		new_stone = rotate_clockwise(self.stone)
		if not check_collision(self.board, new_stone, (self.stone_x, self.stone_y)):
			self.stone = new_stone
	
	def toggle_pause(self):
		self.paused = not self.paused
	
	# def start_game(self):
	# 	if self.gameover:
	# 		self.init_game()
	# 		self.gameover = False

	def set_board(self, board):
		self.board = board

	def get_board(self):
		return self.board
	
	def get_next_stones(self):
		return self.next_stones
	
	def get_movement_bounds(self):
		bounds = []

		bounds.append(config['cols'] - len(self.stone[0]))
		temp_stone = rotate_clockwise(self.stone)
		bounds.append(config['cols'] - len(temp_stone[0]))
		temp_stone = rotate_clockwise(temp_stone)
		bounds.append(config['cols'] - len(temp_stone[0]))
		temp_stone = rotate_clockwise(temp_stone)
		bounds.append(config['cols'] - len(temp_stone[0]))

		return bounds
	
	# def run(self):
	# 	key_actions = {
	# 		'ESCAPE':	self.quit,
	# 		'LEFT':		lambda:self.move(-1),
	# 		'RIGHT':	lambda:self.move(+1),
	# 		'DOWN':		self.hardDrop,
	# 		'UP':		self.rotate_stone,
	# 		'p':		self.toggle_pause,
	# 		'SPACE':	self.start_game
	# 	}
		
	# 	self.gameover = False
	# 	self.paused = False
		
		# pygame.time.set_timer(pygame.USEREVENT+1, config['delay'])
		# clock = pygame.time.Clock()
		# while 1:
		# 	self.screen.fill((0,0,0))
		# 	if self.gameover:
		# 		self.center_msg("""Game Over! Press space to continue""")
		# 	else:
		# 		if self.paused:
		# 			self.center_msg("Paused")
		# 		else:
		# 			self.draw_matrix(self.board, (0,0))
		# 			self.draw_matrix(self.stone, (self.stone_x, self.stone_y))

			# pygame.display.update()
			
			# for event in pygame.event.get():
			# 	if event.type == pygame.USEREVENT+1:
			# 		self.drop()
			# 	elif event.type == pygame.QUIT:
			# 		self.quit()
			# 	elif event.type == pygame.KEYDOWN:
			# 		for key in key_actions:
			# 			if event.key == eval("pygame.K_"+key):
			# 				key_actions[key]()
					
			# clock.tick(config['maxfps'])

# if __name__ == '__main__':
# 	App = TetrisEnv()
# 	App.run()