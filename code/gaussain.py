import torch
import torch.nn.functional as F
class Gaussian: 
    
    def __init__(self, c, mu, sigma, albedo):
        '''
        c: (N, g)
        mu: (N, g, 3)
        sigma: (N, g)
        albedo: (N, g, C)
        N: batch size; g: number of gaussians; C: channel
        '''
        self.c = c
        self.mu = mu
        self.sigma = sigma
        self.albedo = albedo
        
    def get_c(self):
         return self.c
    
    def get_sigma(self):
         return self.sigma
        
    def density(self, x):
        '''
        x: (N, g, H, W, 3)
        return: (N, g, H, W)
        H:height of picture; W: width of picture
        '''
        H = x.size()[2]
        W = x.size()[3]
        
        #c, sigma: (N, g) to (N, g, H, W); mu: (N, g, 3) to (N, g, H, W, 3)
        c = self.get_c().unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        sigma = self.get_sigma().unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        mu = self.mu.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W, 1)
        
        return c * torch.exp(-1 * torch.norm(x - mu, dim = -1) ** 2 / (2 * sigma ** 2))
    
    def project(self, o, n):
        """
        get new gaussian configuration given o and n
        o: (3)
        n: (H, W, 3)
        return: (N, g, H, W) for new_c, new_mu, new_sigma
        """
        
        H = n.size()[0]
        W = n.size()[1]
        
        n = F.normalize(n, p = 2, dim = -1)     
        new_sigma = self.get_sigma().unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        
        #mu - o: (N, g, 3) to (N, g, H, W, 3)
        mu_o = (self.mu - o).unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W, 1)
        new_mu = (mu_o * n).sum(dim = -1)
        
        #c: (N, g) to (N, g, H, W)
        c = self.get_c().unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)
        new_c = c * torch.exp(-1 * (mu_o.norm(dim = -1) ** 2 - new_mu ** 2) / 2 / new_sigma ** 2)
        
        return new_c, new_mu, new_sigma