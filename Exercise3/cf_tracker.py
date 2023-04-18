import numpy as np
import cv2
from ex2_utils import get_patch, extract_histogram, create_epanechnik_kernel, backproject_histogram
from ex3_utils import create_gauss_peak, create_cosine_window
from toolkit_dir.pytracking_toolkit_lite.utils.tracker import Tracker

class CorrelationFilterTracker(Tracker): 
    def initialize(self, image, region):
        self.position = (region[0] + (region[2] // 2), region[1] + (region[3] // 2))
        self.size = ((region[2] // 2 * 2) + 1, (region[3] // 2 * 2) + 1)
        self.patch, mask = get_patch(image, self.position, self.size)
        self.cosine_filter = create_cosine_window(self.size[0], self.size[1])
        self.gauss_filter = create_gauss_peak(self.size, self.parameters.sigma)
        self.gauss_filter_fft = np.fft.fft2(self.gauss_filter)
        self.patch_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.f_hat = np.fft.fft2(self.patch_gray * self.cosine_filter)
        self.f_hat_conjugate = np.conj(self.f_hat)
        self.h_hat_conj = (self.gauss_filter_fft * self.f_hat_conjugate) / (self.f_hat * self.f_hat_conjugate + 0.000001)


    def track(self, image):
        
        patch_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f_hat = np.fft.fft2(patch_gray * self.cosine_filter)
        f_hat_conjugate = np.conj(f_hat)
        h_hat_conj_new = (self.gauss_filter_fft * f_hat_conjugate) / (f_hat * f_hat_conjugate + 0.000001)

        r = np.fft.ifft2(self.h_hat_conj * f_hat)
        r = np.real(r)

        (dx, dy) = np.unravel_index(np.argmax(r, axis=None), r.shape)

    
        (x, y) = self.position
        if dx > self.size[0] // 2:
            dx = dx - self.size[0]

        if dy > self.size[1] // 2:
            dy = dy - self.size[1]

        x_new = x + dx
        y_new = y + dy

        self.position = (x_new, y_new)
        self.h_hat_conj = ((1 - self.parameters.alpha) * self.h_hat_conj) + (self.parameters.alpha * h_hat_conj_new)

        return [self.position[0] - self.size[0] // 2, self.position[1] - self.size[1] // 2, self.size[0], self.size[1]]
class CFParams():
    def __init__(self):
        self.sigma = 1
        self.alpha = 0.1