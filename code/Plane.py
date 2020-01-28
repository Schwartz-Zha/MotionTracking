import torch
class Plane:
    def __init__(self, o, e, e1, e2, l, w, n):
        """
        o: pinhole 
        e: start from pinhole camera o and ends at the centter of Sensor Board
        e1,e2: axis of the board
        l,w: length and width of the board
        n: sample density of all the points in unit length
        """
        
        self.o = o
        self.e = e
        self.e1 = e1
        self.e2 = e2
        self.l = l
        self.w = w
        self.n = n
        self.sample_num = [self.l * self.n, self.w * self.n]
        
    def get_n(self):
        """return sample x, y and the vector from o to each sample point on the plane"""
        
        ori = torch.Tensor(self.l * self.n, self.w * self.n, 3)

        for i in range(0, self.l * self.n):
            for j in range(0, self.w * self.n):
                ori[i, j] = self.e + (i / (self.l * self.n) - 0.5)* self.e1 * self.l + (j / (self.w * self.n) - 0.5)* self.e2 * self.w
                
        return ori
         