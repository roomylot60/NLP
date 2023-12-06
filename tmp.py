n = 17
money = [2,3,5,7]
def solution(n, money):
    answer = [1] + [0]*n
    for coin in money:
        for price in range(coin, n+1):
            if price >= coin:
                answer[price] += answer[price - coin]    
    return answer
print(solution(n,money))