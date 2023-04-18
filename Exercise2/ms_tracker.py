import numpy as np
import cv2
from ex2_utils import Tracker, get_patch, extract_histogram, create_epanechnik_kernel, backproject_histogram

class MeanShiftTracker(Tracker): 
    def initialize(self, image, region):
        self.position = (region[0] + (region[2] // 2), region[1] + (region[3] // 2))
        self.size = ((region[2] // 2 * 2) + 1, (region[3] // 2 * 2) + 1)
        self.patch, mask = get_patch(image, self.position, self.size)
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        self.patch = self.patch.astype("float")
        self.patch[:, :, 0] *= self.kernel
        self.patch[:, :, 1] *= self.kernel
        self.patch[:, :, 2] *= self.kernel
        self.histogram = extract_histogram(self.patch, self.parameters.nbins)  

    def track(self, image):
        go = True
        cnt = 0

        a = np.arange(-(self.size[0] // 2), (self.size[0] // 2), 1)  
        b = np.arange(-(self.size[1] // 2), (self.size[1] // 2))  
        xi, xi = np.meshgrid(a, b)  
        
        while go:
            cnt += 1
            patch, mask = get_patch(image, self.position, self.size)
            patch = patch.astype("float")
            patch[:, :, 0] *= self.kernel
            patch[:, :, 1] *= self.kernel
            patch[:, :, 2] *= self.kernel

            self.p = extract_histogram(patch, self.parameters.nbins)
            w = np.sqrt(self.histogram / (p + 0.001))

            w = backproject_histogram(patch, w, self.parameters.nbins)

            

            (x, y) = self.position
            xnew = x + np.sum(xi*w) / np.sum(w)
            ynew = y + np.sum(yi*w) / np.sum(w)

            self.position = (xnew, ynew)
            if cnt == self.parameters.steps:         
                break
        self.histogram = ((1 - self.parameters.alpha) * self.histogram) + (self.parameters.alpha * self.p)
    
        return [self.position[0] - self.size[0] // 2, self.position[1] - self.size[1] // 2, self.size[0], self.size[1]]
class MSParams():
    def __init__(self):
        self.nbins = 16
        self.sigma = 1
        self.steps = 25
        self.alpha = 0.5
