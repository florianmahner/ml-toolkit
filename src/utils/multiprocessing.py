from joblib import Parallel, delayed


def submit_parallel_jobs(
    func, args, joblib_kwargs: dict = {"n_jobs": -1, "verbose": 10}
):
    """
    Submit parallel jobs to a function.
    """
    parallel = Parallel(**joblib_kwargs)
    return parallel(delayed(func)(*arg) for arg in args)
