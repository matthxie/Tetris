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
	
	[[0, 1, 1],
	 [1, 1, 0]],
	
	[[1, 1, 0],
	 [0, 1, 1]],
	
	[[1, 0, 0],
	 [1, 1, 1]],
	
	[[0, 0, 1],
	 [1, 1, 1]],
	
	[[1, 1, 1, 1]],
	
	[[1, 1],
	 [1, 1]]
]

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
	# del board[row]
	board = np.delete(board, row, axis=0)
	board = np.vstack((np.zeros((1, 10), dtype=np.float32), board))

	return board
	
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
	
	return np.array(board)

def get_board_heights(board):
	boardHeights = []

	for i in range(10):
		for j in range(20):
			if board[j][i] != 0:
				boardHeights.append(20-j)
				break
			if j == 19 and board[j][i] == 0:
				boardHeights.append(0)

	return boardHeights

def get_bumpiness(heights):
	currs = np.array(heights[:-1])
	nexts = np.array(heights[1:])
	diffs = np.abs(currs - nexts)
	total_bumpiness = np.sum(diffs)

	return total_bumpiness

def get_num_holes(board):
	holes = 0

	for i in range(10):
		new_col = True
		for j in range(20):
			if new_col and board[j][i] != 0:
				new_col = False
				continue
			if board[j][i] == 0 and not new_col:
				holes += 1

	return holes

class TetrisEnv():
	def __init__(self, state_dim=11, action_dim=(10,20)):
		pygame.init()
		# pygame.key.set_repeat(250,25)

		self.gameover = 0
		self.paused = False
		self.blocks_placed = 0
		self.lines_cleared = 0
		
		self.width = config['cell_size']*config['cols']
		self.height = config['cell_size']*config['rows']
		self.next_stones = []
		self.stone_orientation = 0

		for _ in range(3):
			self.next_stones.append(rand(1, len(tetris_shapes)+1))

		self.board = new_board()
		
		pygame.event.set_blocked(pygame.MOUSEMOTION)

		# self.init_game()

	def calc_reward(self, board):
		reward = 0.0
		
		heights = get_board_heights(board)
		holes = get_num_holes(board)
		bumpiness = get_bumpiness(heights)

		reward += bumpiness * -0.4
		reward += sum(heights) * -0.51
		reward += np.sum(np.abs(np.diff(heights))) * -0.18
		reward += holes * -0.36

		return reward
	
	def check_invalid_move(self, x_dest, r_dest=-1):
		stone = self.next_stones[0]
		if r_dest == -1:
			r_dest = self.stone_orientation

		if stone == 7:
			if x_dest == 9:
				# print(self.stone, ", ", rot, ", ", stone, ", ", x_dest, ", ", r_dest)
				return True
		elif stone == 6:
			if (r_dest == 0 or r_dest == 2) and x_dest > 6:
				# print(self.stone, ", ", rot, ", ", stone, ", ", x_dest, ", ", r_dest)
				return True
		else:
			if x_dest == 9:
				# print(self.stone, ", ", rot, ", ", stone, ", ", x_dest, ", ", r_dest)
				return True
			elif x_dest == 8 and r_dest != 1 and r_dest != 3:
				# print(self.stone, ", ", rot, ", ", stone, ", ", x_dest, ", ", r_dest)
				return True
			
		return False

	def reset(self):
		self.init_game()

		heights = get_board_heights(self.board)
		holes = get_num_holes(self.board)
		bumpiness = 0

		result_info = np.array(self.next_stones, dtype=np.int64)
		result_info = np.append(result_info, sum(heights))
		result_info = np.append(result_info, bumpiness)
		result_info = np.append(result_info, holes)

		next_state = self.board[:-1].flatten()
		next_state = np.concatenate((next_state.flatten(), result_info))
		next_state = next_state.tolist()

		return next_state

	def step(self, x_dest, r_dest, probe, display=False):
		board_backup = self.board.copy()
		next_stones_backup = self.next_stones.copy()
		stone_backup = self.stone.copy()
		stone_x_backup = self.stone_x
		stone_y_backup = self.stone_y
		stone_orientation_backup = self.stone_orientation

		for i in range(r_dest):
			self.rotate_stone()

		self.blocks_placed += 1
		old_reward = self.calc_reward(self.board)

		results = self.move_to_placement(x_dest, probe)
		lines_cleared = results[0]
		result_board = results[1]
		self.lines_cleared += lines_cleared

		if not probe:
			self.board = result_board

		heights = get_board_heights(result_board)
		bumpiness = get_bumpiness(heights)
		holes = get_num_holes(result_board)
		reward = 1 + 10*lines_cleared**2
		# reward = self.calc_reward(result_board) - old_reward + 10*lines_cleared

		next_state = self.board[:-1]
		result_info = np.array(self.next_stones, dtype=np.int64)
		result_info = np.append(result_info, np.sum(heights))
		result_info = np.append(result_info, bumpiness)
		result_info = np.append(result_info, holes)

		if display:
			# print(self.calc_reward(result_board), ", ", old_reward)
			print(result_board)

			if lines_cleared > 0:
				print("lines cleared: ", lines_cleared)

		if probe:
			self.board = board_backup
			self.stone = stone_backup
			self.next_stones = next_stones_backup
			self.stone_x = stone_x_backup
			self.stone_y = stone_y_backup
			self.stone_orientation = stone_orientation_backup

		next_state = np.concatenate((next_state.flatten(), result_info))
		next_state = next_state.tolist()

		return next_state, reward, self.gameover, lines_cleared

	def render(self):
		pass

	def new_stone(self, probe=True):
		self.next_stones.append(rand(1, len(tetris_shapes)+1))

		self.stone = tetris_shapes[self.next_stones[1]-1]
		self.next_stones.pop(0)
		
		self.stone_x = int(config['cols'] / 2 - len(self.stone[0])/2)
		self.stone_y = 0

		if probe: return

		for col in self.board[0]:
			if col == 1:
				self.gameover = 1
				break
		# if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
		# 	self.gameover = 1
	
	def init_game(self):
		self.gameover = 0
		self.paused = False
		self.blocks_placed = 0
		self.lines_cleared = 0

		self.next_stones =[rand(1, len(tetris_shapes)+1)]
		self.next_stones.append(rand(1, len(tetris_shapes)+1))
		self.next_stones.append(rand(1, len(tetris_shapes)+1))

		self.board = new_board()
		self.new_stone(probe=False)
		self.stone_orientation = 0

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
	
	def move_to_placement(self, x_dest, probe=True):
		delta_x = x_dest - self.stone_x
		# self.toggle_pause()

		self.move(delta_x)
		
		lines_cleared = self.hardDrop(probe)

		# result_board = np.array(self.board)[:20, :]
		result_board = np.array(self.board, dtype=np.float32)

		# self.toggle_pause()

		return [lines_cleared, self.board]
	
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
	
	def drop(self, probe=True):
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
	
	def hardDrop(self, probe=True):
		lines_cleared = 0

		# if not self.gameover and not self.paused:
		while not check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
			self.stone_y += 1
		self.board = join_matrixes(self.board, self.stone, (self.stone_x, self.stone_y))
		self.new_stone(probe)
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
	# 		self.gameover = 0

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
	
	def get_next_states(self):
		obses = []
		actions = []
		rewards = []
		bounds = self.get_movement_bounds()

		for r in range(len(bounds)):
			for x in range(bounds[r]+1):
				new_obs, reward, done, info = self.step(x, r, probe=True, display=False)
				obses.append(new_obs)
				actions.append(r*10 + x)
				rewards.append(reward)
		return [obses, actions]
		
		
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