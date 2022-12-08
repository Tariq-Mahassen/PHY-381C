import necessary modules
import numpy as np

class AbelianSandpile:
    
    def __init__(self, n = 100, random_state = None):
        self.n = n
        self.grid = np.random.choice([0,1,2,3], size(n,n)) #creates an nxn array of integers chosen randomly from the list [0,1,2,3]
        self.history = self.grid.copy() #copies the grid so that we can compare the final state to the initial
        self.topples = [] #tracks topples


    def add_and_topple(self, i, j):

        self.grid[i, j] += 1

        if self.grid[i, j] < 4:
            return None

        else:
            # Topple high site
            self.grid[i, j] -= 4

            # Implement the absorbing boundary conditions; sandgrains that fall off the edge of the grid are lost.
            if i > 0:
                self.add_and_topple(i - 1, j)
            if i < self.n - 1:
                self.add_and_topple(i + 1, j)
            if j > 0:
                self.add_and_topple(i, j - 1)
            if j < self.n - 1:
                self.add_and_topple(i, j + 1)
            return None
           
def step():
    #drop a grain
    #solve for topple events until sanple stabilises
    
    x,y = np.random.choice(self.n, 2) #returns an array of 2 elements and assigns the first column to x and the second to y
    
    self.add_and_topple(x,y) #drops a sand grain at position (x,y) on the grid and begins toppling sequence
    
    topples = 0 #counts number of topples until grid is stabilised
    while np.any(self.grid >= 4): #while any grid has squares greater than 4, topple, np.any returns True if condition is met.
        topplesites = np.where(self.grid >= 4) #returns a tuple of 2 arrays, topplesites[0] is the column and topplesites[1] is the row
        for i in range(len(topplesitees[0])):
            ii = topplesites[0][i]
            jj = topplesites[1][i]
            
            self.grid(ii, jj) -= 4
            
            if ii > 0:
                self.grid[ii - 1, jj] += 1
            if ii < self.n - 1:
                self.grid[ii + 1, jj] += 1
            if jj > 0:
                self.grid[ii, jj - 1] += 1
            if jj < self.n - 1:
                self.grid[ii, jj + 1] += 1
            topples += 1
        if topples > 0:
            self.topples.append(topples)
            
def check_difference(grid1, grid2):
        return np.sum(grid1 != grid2) #boolean sum

def simulate(self, n_steps):
    l = []
    for i in range(n_steps):
        self.step()
        if self.site_difference(self.grid, self.grid.copy()) > 0:
            return(self.grid)
