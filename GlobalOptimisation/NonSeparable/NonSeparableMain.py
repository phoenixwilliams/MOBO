

if __name__ == "__main__":
    import random
    import Functions
    from OptimizationProblems.SingleObjective import NonSeperable

    problem1 = Functions.Rosenbrock({"bounds": [-5, 10]}, 1, 100)
    problem2 = NonSeperable.rosenbrock
    arr = [1 for _ in range(1000)]

    print(problem1(arr))
    print(problem2(arr))