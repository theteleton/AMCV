import numpy as np
from ex2_utils import generate_responses_1, get_patch, generate_responses_2

def mode(image, starting_point, kernel_size, kernel, terminate_criteria):
    go = True
    (y, x) = starting_point 
    a = np.arange(-(self.size[0] // 2), (self.size[0] // 2), 1)  
    b = np.arange(-(self.size[1] // 2), (self.size[1] // 2))  
    xi, xi = np.meshgrid(a, b)  

    steps = 0
    while go:
        steps += 1
        patch, mask = get_patch(image, (x, y), kernel_size)
        if np.sum(patch) == 0:
            print("All numbers in the area are zeros, find another point")
            break
        if steps == 10000:
            print("too much steps")
        xnew = x + np.sum(xi*patch*kernel) / np.sum(patch*kernel)
        ynew = y + np.sum(yi*patch*kernel) / np.sum(patch*kernel)
        if x == xnew and y == ynew:
            print("A flat point, maybe local maximum maybe not")
            return (int(y), int(x)), steps
        if np.sqrt((xnew - x) ** 2 + (y - ynew) ** 2) <= terminate_criteria:
            print("Finished")
            return (int(y), int(x)), steps
        x = xnew
        y = ynew



if __name__ == "__main__":
    #image = generate_responses_1()
    image = generate_responses_2()
    print(np.argmax(image))
    print(mode(image, (40, 50), (7, 7), np.ones([7, 7]), 0.1))
