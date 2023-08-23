import torch

paint_table = {
        0: [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],

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

        10: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        11: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        12: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        13: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        14: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        15: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        16: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        17: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

        18: [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]],

}

move_table = {
    0:  (0, 0),
    1:  (0, 0),
    2:  (0, 1),
    3:  (1, 0),
    4:  (0, -1),
    5:  (-1, 0),
    6:  (1, 1),
    7:  (1, -1),
    8:  (-1, -1),
    9:  (-1, 1),
    10: (0, 0),
    11: (0, 1),
    12: (1, 0),
    13: (0, -1),
    14: (-1, 0),
    15: (1, 1),
    16: (1, -1),
    17: (-1, -1),
    18: (-1, 1),
}



def convert(strokes, image_size):
    image = torch.ones(image_size, image_size)

    #noise = torch.randn_like(image) * 0.005
    #image = image + noise

    depth = -1

    x = image_size // 2
    y = image_size // 2

    for word in strokes:
        (ax, ay) = move_table[word]

        check_range = lambda val, delta, size:  1 <= val+delta < size - 1
        tfs = [check_range(val, delta, image_size) for val, delta in zip((x, y), (ax, ay))]
        if all(tfs):
            x += ax
            y += ay
            image[x-1:x+1+1, y-1:y+1+1] += (torch.tensor(paint_table[word]) * depth)
            image = torch.clamp(image, max=1, min=0)
            #image[x-1:x+1+1, y-1:y+1+1] += (torch.tensor(paint_table[10]) * 30)

    return image


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

        dotimg = torch.tensor(paint_table[word]).to(device)
        canvas[rx-1:rx+1+1, ry-1:ry+1+1] += dotimg * depth
        canvas = torch.clamp(canvas, max=0, min=-1)

        #image[x-1:x+1+1, y-1:y+1+1] += (torch.tensor(paint_table[10]) * 30)

    return rx, ry, canvas

