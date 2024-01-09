def solution(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
            
nums = [1,3,7,4]
target = 7
print(solution(nums, target))