#grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]

class Solution:
    def countNegatives(self, grid: [[int]] ) -> int:
        neg = 0
        for i in grid:
            for j in i:
                if j < 0:
                    neg += 1 
        return neg
        


sol = Solution()
grid = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]
print(sol.countNegatives(grid=grid))