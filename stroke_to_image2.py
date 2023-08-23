import torch

N_WORD = 10

brush = {
    0: [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]],
    1: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    2: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    3: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    4: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    5: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    6: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    7: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    8: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
    9: [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],
}

move_table = {
    0:  (0, 0),
    1:  (0, 0),
    2:  (0, 1),
    3:  (1, 1),
    4:  (1, 0),
    5:  (1, -1),
    6:  (0, -1),
    7:  (-1, -1),
    8:  (-1, 0),
    9:  (-1, 1),
}

def draw(canvas, word, pos, image_size, device):
    depth = -1
    (dx, dy) = move_table[word]

    rx, ry = pos[0], pos[1]

    ## Check if the cursor position is inside of the canvas.
    ## Because Brush size is 3x3, range check is 1 <= current pos < size - 1
    is_inner = lambda pos, delta, size:  1 <= pos+delta < size - 1

    ## Check if the brush position is inside or not for all dimension
    tfs = [is_inner(pos, delta, image_size) for pos, delta in zip((rx, ry), (dx, dy))]
    if all(tfs):
        rx += dx
        ry += dy

        dotimg = torch.tensor(brush[word]).to('cpu')
        canvas[rx-1:rx+1+1, ry-1:ry+1+1] += dotimg * depth
        canvas = torch.clamp(canvas, max=0, min=-1)

    return rx, ry, canvas

