# from rembg import remove
# import cv2

# img = cv2.imread('./ccb2.png')
# rmvd_img = remove(img)
# cv2.imwrite('./rmvd.png', rmvd_img)


def solution(s):
    stack = []
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    return ''.join(stack)

print(solution('abbaca'))

