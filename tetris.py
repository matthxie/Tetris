import numpy as np
from random import randrange as rand

config = {"cell_size": 20, "cols": 10, "rows": 20, "delay": 750, "maxfps": 30}

colors = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 150, 0),
    (0, 0, 255),
    (255, 120, 0),
    (255, 255, 0),
    (180, 0, 255),
    (0, 220, 220),
]

tetris_shapes = [
    np.array([[1, 1, 1], [0, 1, 0]]),
    np.array([[0, 1, 1], [1, 1, 0]]),
    np.array([[1, 1, 0], [0, 1, 1]]),
    np.array([[1, 0, 0], [1, 1, 1]]),
    np.array([[0, 0, 1], [1, 1, 1]]),
    np.array([[1, 1, 1, 1]]),
    np.array([[1, 1], [1, 1]]),
]


def rotate_clockwise(shape):
    return np.transpose(np.flip(shape, 1))


def check_collision(board, shape, offset):
    off_x, off_y = offset
    shape_h, shape_w = shape.shape

    if (
        off_x < 0
        or off_y < 0
        or off_x + shape_w > config["cols"]
        or off_y + shape_h > config["rows"] + 1
    ):
        return True

    board_section = board[off_y : off_y + shape_h, off_x : off_x + shape_w]
    if board_section.shape != shape.shape:
        return True

    return np.any(np.logical_and(board_section, shape))


def remove_row(board, row):
    board[1 : row + 1] = board[0:row]
    board[0] = 0
    return board


def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1


def new_board():
    board = np.zeros((config["rows"] + 1, config["cols"]), dtype=np.float32)
    board[-1] = 1
    return board


def get_board_heights(board):
    heights = np.zeros(config["cols"], dtype=np.int32)

    for col in range(config["cols"]):
        filled_cells = np.where(board[: config["rows"], col] > 0)[0]
        if filled_cells.size > 0:
            heights[col] = config["rows"] - filled_cells[0]

    return heights


def get_bumpiness(heights):
    diffs = np.abs(heights[1:] - heights[:-1])

    return np.sum(diffs)


def get_num_holes(board):
    first_one_mask = np.zeros_like(board, dtype=bool)
    first_one_indices = np.argmax(board == 1, axis=0)

    valid_columns = first_one_indices < board.shape[0]

    row_indices = first_one_indices[valid_columns]
    col_indices = np.where(valid_columns)[0]

    first_one_mask[row_indices, col_indices] = True

    below_mask = np.zeros_like(board, dtype=bool)
    for col in range(board.shape[1]):
        if valid_columns[col]:
            below_mask[first_one_indices[col] + 1 :, col] = True

    holes = np.sum((board == 0) & below_mask)

    return holes


class TetrisEnv:
    def __init__(self, state_dim=11, action_dim=(10, 20)):
        self.gameover = 0
        self.paused = False

        self.width = config["cell_size"] * config["cols"]
        self.height = config["cell_size"] * config["rows"]
        self.next_stones = []
        self.num_holes = 0
        self.bumpiness = 0
        self.stone_orientation = 0

        for _ in range(3):
            self.next_stones.append(rand(1, len(tetris_shapes) + 1))

        self.board = new_board()

    def check_invalid_move(self, x_dest, r_dest=-1):
        stone = self.next_stones[0]
        if r_dest == -1:
            r_dest = self.stone_orientation

        if stone == 7:
            if x_dest == 9:
                return True
        elif stone == 6:
            if (r_dest == 0 or r_dest == 2) and x_dest > 6:
                return True
        else:
            if x_dest == 9:
                return True
            elif x_dest == 8 and r_dest != 1 and r_dest != 3:
                return True

        return False

    def reset(self):
        self.init_game()

        heights = get_board_heights(self.board)
        self.num_holes = 0
        self.bumpiness = 0

        result_info = np.array(self.next_stones, dtype=np.int64)
        result_info = np.append(result_info, sum(heights))
        result_info = np.append(result_info, self.bumpiness)
        result_info = np.append(result_info, self.num_holes)

        next_state = self.board[:-1].flatten()
        next_state = np.concatenate((next_state.flatten(), result_info))
        next_state = next_state.tolist()

        return next_state

    def step(self, x_dest, r_dest, probe, display=False):
        for i in range(r_dest):
            self.rotate_stone()

        results = self.move_to_placement(x_dest, probe)
        lines_cleared = results[0]
        self.board = results[1]

        heights = get_board_heights(self.board)
        bumpiness = get_bumpiness(heights)
        holes = get_num_holes(self.board)
        delta_holes = holes - self.num_holes
        delta_bumpiness = bumpiness - self.bumpiness
        reward = 50 * lines_cleared**2 - delta_holes - delta_bumpiness
        self.num_holes = holes
        self.bumpiness = bumpiness

        next_state = self.board[:-1]
        result_info = np.array(self.next_stones, dtype=np.int64)
        result_info = np.append(result_info, np.sum(heights))
        result_info = np.append(result_info, delta_bumpiness)
        result_info = np.append(result_info, delta_holes)

        next_state = np.concatenate((next_state.flatten(), result_info))
        next_state = next_state.tolist()

        return (
            next_state,
            reward,
            self.gameover,
            lines_cleared,
            delta_holes,
            delta_bumpiness,
        )

    def render(self):
        pass

    def new_stone(self, probe=True):
        self.next_stones.append(rand(1, len(tetris_shapes) + 1))

        self.stone = tetris_shapes[self.next_stones[1] - 1]
        self.next_stones.pop(0)

        self.stone_x = int(config["cols"] / 2 - len(self.stone[0]) / 2)
        self.stone_y = 0

        if probe:
            return

        for col in self.board[0]:
            if col == 1:
                self.gameover = 1
                break

    def init_game(self):
        self.gameover = 0
        self.paused = False

        self.next_stones = [rand(1, len(tetris_shapes) + 1)]
        self.next_stones.append(rand(1, len(tetris_shapes) + 1))
        self.next_stones.append(rand(1, len(tetris_shapes) + 1))

        self.board = new_board()
        self.new_stone(probe=False)
        self.stone_orientation = 0

        self.num_holes = 0
        self.bumpiness = 0

    def move_to_placement(self, x_dest, probe=True):
        delta_x = x_dest - self.stone_x

        self.move(delta_x)

        lines_cleared = self.hardDrop(probe)

        return [lines_cleared, self.board]

    def move(self, delta_x):
        new_x = self.stone_x + delta_x
        if new_x < 0:
            new_x = 0
        if new_x > config["cols"] - len(self.stone[0]):
            new_x = config["cols"] - len(self.stone[0])
        if not check_collision(self.board, self.stone, (new_x, self.stone_y)):
            self.stone_x = new_x

    def drop(self, probe=True):
        self.stone_y += 1
        if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.board = join_matrixes(
                self.board, self.stone, (self.stone_x, self.stone_y)
            )
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

    def set_board(self, board):
        self.board = board

    def get_board(self):
        return self.board

    def get_next_stones(self):
        return self.next_stones

    def get_state(self):
        heights = get_board_heights(self.board)
        bumpiness = get_bumpiness(heights)
        holes = get_num_holes(self.board)

        state = self.board[:-1]
        result_info = np.array(self.next_stones, dtype=np.int64)
        result_info = np.append(result_info, np.sum(heights))
        result_info = np.append(result_info, bumpiness)
        result_info = np.append(result_info, holes)

        state = np.concatenate((state.flatten(), result_info))
        state = state.tolist()

        return (state, result_info, self.next_stones)

    def get_movement_bounds(self):
        bounds = []

        stone = self.stone
        for _ in range(4):
            bounds.append(config["cols"] - stone.shape[1])
            stone = rotate_clockwise(stone)

        return bounds

    def get_invalid_moves(self):
        bounds = self.get_movement_bounds()
        mask = []

        for i in range(len(bounds)):
            mask.extend([True] * bounds[i] + [False] * (10 - bounds[i]))

        return mask
