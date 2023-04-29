import numpy as np
import cv2
import time
from ex2_utils import get_patch, extract_histogram, create_epanechnik_kernel, backproject_histogram
from ex3_utils import create_gauss_peak, create_cosine_window
from ex4_utils import sample_gauss
from kalman import compute_matrices
from utils.tracker import Tracker


class ParticleFilter(Tracker): 

    def __init__(self):
        super().__init__()
        self.parameters = PFParams()

    def name(self):
        return "particle--21"
    
    def initialize(self, image, region):
        self.n = self.parameters.n
        self.position = (region[0] + (region[2] // 2), region[1] + (region[3] // 2))
        self.size = ((region[2] // 2 * 2) + 1, (region[3] // 2 * 2) + 1)
        self.patch, mask = get_patch(image, self.position, self.size)
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)
        self.patch = self.patch.astype("float")
        self.patch[:, :, 0] *= self.kernel
        self.patch[:, :, 1] *= self.kernel
        self.patch[:, :, 2] *= self.kernel
        self.histogram = extract_histogram(self.patch, self.parameters.nbins)
        self.F, self.Q, _, _ = compute_matrices(self.parameters.model, 1, self.parameters.q * min(self.size[0], self.size[1]), 1)
        self.particles = np.zeros((self.n, self.F.shape[0]))
        self.weights = np.ones((self.n))
        self.particles[:, :2] = sample_gauss(self.position, self.Q[:2, :2], self.n)



    def track(self, image):
        particles = self.particles
        weights = self.weights

        weights_norm = weights / np.sum(weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.n, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        particles_new = particles[sampled_idxs.flatten(), :]

        for i in range(self.n):
            self.particles[i, :] = np.transpose(np.dot(self.F, particles_new[i, :].T)) + sample_gauss(np.zeros(self.F.shape[0]), self.Q, 1)
        
        new_weights = self.weights
        for i in range(self.n):
            new_particle = (self.particles[i, 0], self.particles[i, 1])
            if new_particle[0] < 0 or new_particle[1] < 0 or new_particle[0] >= image.shape[1] or new_particle[1] >= image.shape[0]:
                new_weights[i] = 0
            else:
                patch, mask = get_patch(image, new_particle, self.size)
                patch = patch.astype("float")
                patch[:, :, 0] *= self.kernel
                patch[:, :, 1] *= self.kernel
                patch[:, :, 2] *= self.kernel

                new_hist = extract_histogram(patch, self.parameters.nbins)
                old_hist = self.histogram
                hellinger_distance = np.linalg.norm(np.sqrt(new_hist / np.sum(new_hist)) - np.sqrt(old_hist / np.sum(old_hist))) / np.sqrt(2)
                new_weights[i] = np.exp(-(hellinger_distance ** 2) /(2 * (self.parameters.sigma2 ** 2)))



        self.weights = new_weights
        new_position = [0, 0]
        for i in range(self.n):
            
            new_position[0] = new_position[0] + (self.weights[i] * self.particles[i, 0])
            new_position[1] = new_position[1] + (self.weights[i] * self.particles[i, 1])

        
        new_position[0] /= np.sum(weights)
        new_position[1] /= np.sum(weights)


        self.position = (int(new_position[0]), int(new_position[1]))

        patch, mask = get_patch(image, self.position, self.size)
        patch = patch.astype("float")
        patch[:, :, 0] *= self.kernel
        patch[:, :, 1] *= self.kernel
        patch[:, :, 2] *= self.kernel

        self.p = extract_histogram(patch, self.parameters.nbins)
        self.histogram = ((1 - self.parameters.alpha) * self.histogram) + (self.parameters.alpha * self.p)
        return [self.position[0] - self.size[0] // 2, self.position[1] - self.size[1] // 2, self.size[0], self.size[1]]
    


 
class PFParams():
    def __init__(self):
        self.n = 100
        self.nbins = 16
        self.q = 0.75
        self.sigma = 1
        self.sigma2 = 0.1
        self.alpha = 0.05
        self.model = "NCV"