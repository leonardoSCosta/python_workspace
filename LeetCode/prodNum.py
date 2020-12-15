class ProductOfNumbers:
    numList = []
    lenL = 0
    def __init__(self):
        self.numList = []
        self.lenL = 0
    def add(self, num: int) -> None:
        self.numList.append(num)
        self.lenL += 1

    def getProduct(self, k: int) -> int:
        prod = self.numList[self.lenL-1]
        if k == 1:
            return prod
            
        for n in self.numList[self.lenL-k:-1]:
            prod = n * prod
        return prod
        


# Your ProductOfNumbers object will be instantiated and called as such:
obj = ProductOfNumbers()
obj.add(5)
obj.add(3)
obj.add(2)
obj.add(4)
param_2 = obj.getProduct(3)
print(param_2)