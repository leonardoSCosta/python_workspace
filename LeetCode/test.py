class Solution:
    def twoSum(self, nums: [int], target: int) -> [int]:

        for num,i in zip(nums,range(0,len(nums))):
            for secNum,j in zip(nums[i+1:],range(i+1,len(nums))):
                if secNum + num == target:
                    return [i,j]

    def findMedianSortedArrays(self, nums1: [int], nums2: [int]) -> float:
        nums = sorted(nums1 + nums2)
        
        rest = len(nums)%2
        numMid = int(len(nums)/2)
        print(rest, numMid, nums)
        if rest < 1:
            return (nums[numMid]+nums[numMid+1])/2
        else:
            return nums[numMid]
            
sol = Solution()
print(sol.findMedianSortedArrays([],[1,2,3,4,5]))