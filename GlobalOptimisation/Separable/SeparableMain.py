




if __name__ == "__main__":
    import random
    import Functions
    from OptimizationProblems.SingleObjective import Separable


    problem1 = Functions.Styblinski_tank({"bounds":[-5, 5]})
    problem2 = Separable.styblinski_tank
    arr = [-2.896593979760246 for _ in range(1000)]

    print(problem1(arr))
    print(problem2(arr))




