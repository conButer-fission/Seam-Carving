import re
import numpy   as np
import skimage as ski
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def read_image(path : str) -> np.ndarray:
    image =  ski.io.imread(path)
    # drop the alpha channel
    image = image[:,:,0 : 3]
    # print(image.shape)
    return image

def save_image(path : str, new_image : np.ndarray) -> None:
    ski.io.imsave(path, new_image, quality=90)
    


def compute_gradient(image : np.ndarray) -> np.ndarray:
    
    # to avoid flops
    dx = np.array([[1, 0, -1],
                    [2, 0 , -2],
                    [1, 0, -1]])
    dy = dx.T
    
    # Replicate for R G B
    dx = np.stack(3 * [dx], axis = 2)
    dy = np.stack(3 * [dy], axis = 2)
    gradient_sum = np.absolute(convolve(image, dx)) + np.absolute(convolve(image, dy))
    gradient_sum = gradient_sum.sum(axis=2)
    return gradient_sum
def get_min(image : np.ndarray) -> np.ndarray:
    gradient_sum = compute_gradient(image)
    r, c, __ = np.shape(image)

    remove = np.zeros((r, c), dtype=bool)
    min_path = np.zeros((r, c), dtype=np.int32)
    prev_c = np.zeros((r, c), dtype=np.int32)
    
    min_path[0] = gradient_sum[0, 0 : c]
    for i in range(1, r):
        for j in range(0, c):
            min_el = 0
            if j > 0:
                prev_c[i, j] = j - 1 + np.argmin(min_path[i - 1, j - 1 : j + 2])
                min_el = min(min_path[i - 1, j - 1 : j + 2])
            else:
                prev_c[i, j] = j + np.argmin(min_path[i - 1, j: j + 2])
                min_el = min(min_path[i - 1, j: j + 2])
                
            min_path[i, j] = gradient_sum[i, j] + min_el
    
    c_c = np.argmin(min_path[-1])
    for c_r in range(r - 1, -1, -1):
        remove[c_r, c_c] = 1
        c_c = prev_c[c_r, c_c]
    return remove    
 
def column_remove(image : np.ndarray, seam_idx : list, to_mark : list, mm : int):
    remove = get_min(image)
    # print(image.shape)
    new_image = np.zeros((image.shape[0], image.shape[1] - 1, image.shape[2]), dtype=np.uint8)
    remove_r, remove_c = np.shape(remove)
    for r in range(0, remove_r):
        n_c = 0
        for c in range(remove_c - 1, -1, -1):
            if(not remove[r, c]):
                new_image[r, n_c] = image[r, c]
                n_c+=1
            else:
                if(type(seam_idx) == np.ndarray):
                    seam_idx = [list(row) for row in seam_idx]
                to_mark[mm].append(seam_idx[r][c])
                del seam_idx[r][c]
    return new_image, seam_idx, to_mark


def construct_visual(image : np.ndarray, to_mark : list) -> np.ndarray:
    horizontal = image
    vertical = image.copy()
    for i in to_mark[0]:
        horizontal[i[0], i[1]] = [255, 0, 0]
    for i in to_mark[1]:
        vertical[i[0], i[1]] = [255, 0, 0]
    return horizontal, vertical

def scale_image(path : str, new_width : int, new_height : int):
    old_image = read_image(path)
    visual_image = old_image
    to_mark = [[], []]
    old_height, old_width, _ = np.shape(old_image)
    
    seam_idx = [old_width * [(1, 1)] for _ in range(old_height)]
    for i in range(0, old_height):
        for j in range(0, old_width):
            seam_idx[i][j] = (i, j)

    if(new_width >= old_width or new_height >= old_height):
        raise ValueError("new width/height should be less than original one")
    
    
    dx, dy = [old_width - new_width, old_height - new_height]
    new_image = old_image
    for __ in range(0, dx):
        new_image, seam_idx, to_mark = column_remove(new_image, seam_idx, to_mark, 0)

    # horizontal removal
    new_image = np.rot90(new_image, k = 1, axes=(0, 1))
    seam_idx = np.rot90(seam_idx, k = 1, axes=(0, 1))
    for __ in range(0, dy):
        new_image, seam_idx, to_mark = column_remove(new_image, seam_idx, to_mark, 1)

    # no idea why k = (1, 0) works instead of (0, 1)!
    new_image = np.rot90(new_image, k = 3, axes=(1, 0))
    verti, horiz = construct_visual(visual_image, to_mark)
    save_image(f'{re.sub('\..{1,}', "", path)}_scaled{re.findall('\..{1,}', path)[0]}', new_image)
    save_image(f'{re.sub('\..{1,}', "", path)}_visual_vert{re.findall('\..{1,}', path)[0]}', verti)
    save_image(f'{re.sub('\..{1,}', "", path)}_visual_horiz{re.findall('\..{1,}', path)[0]}', horiz)
    
    
    
if __name__ == '__main__':
    scale_image("test", 200, 100)