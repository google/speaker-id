"""Script to estimate FLOPs of multi-stage clustering."""
import os
import copy
import csv
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pypapi import events, papi_high as high
import spectralcluster


INF = float("inf")


def run_flops_spectral(N, D, autotune=True):
    """Estimate FLOPs of spectral clustering algorithm.

    Args:
        N: number of inputs
        D: input dimension
        autotune: whether to use autotune or not,
            see https://arxiv.org/abs/2003.02405

    Returns:
        FLOPs
    """
    if N == 0:
        return 0

    X = np.random.rand(N, D)
    clusterer = copy.deepcopy(spectralcluster.configs.turntodiarize_clusterer)
    if autotune:
        clusterer.autotune.p_percentile_min = 0.8
        clusterer.autotune.p_percentile_max = 0.95
        clusterer.autotune.init_search_step = 0.01
    else:
        clusterer.autotune = None
    high.start_counters([events.PAPI_FP_OPS, ])
    clusterer.predict(X)
    flops = high.stop_counters()
    return flops[0]


def run_flops_ahc(N, stop, D):
    """Estimate FLOPs of AHC algorithm.

    Args:
        N: number of inputs
        stop: AHC stops when reduced to this many clusters
        D: input dimension

    Returns:
        FLOPs
    """
    if N == 0:
        return 0

    X = np.random.rand(N, D)
    AHC = AgglomerativeClustering(
        n_clusters=stop,
        affinity="cosine",
        linkage="complete")

    high.start_counters([events.PAPI_FP_OPS, ])
    AHC.fit(X)
    flops = high.stop_counters()
    return flops[0]


def run_flops(N, D=256, L=INF, U1=INF, U2=INF, autotune=True):
    """Estimate the flops for N inputs.

    Args:
        N: number of inputs
        D: input dimension
        L: lower bound of main clusterer
        U1: upper bound of main clusterer
        U2: upper bound of pre-clusterer
        autotune: whether spectral clusterer uses autotune

    Returns:
        A CSV row
    """
    print("Estimating FLOPs, N =", N, ", D =", D, ", L =", L,
          ", U1 =", U1, ", U2 =", U2, ", autotune =", autotune)
    fallback_flops = 0
    main_flops = 0
    pre_flops = 0
    if N < L:
        fallback_flops = run_flops_ahc(N, 2, D)
    else:
        main_clusterer_in = N
        pre_clusterer_in = 0
        pre_clusterer_out = 0
        if N >= U1:
            main_clusterer_in = U1
            pre_clusterer_in = N
            pre_clusterer_out = U1
        if N >= U2:
            pre_clusterer_in = U2

        main_flops = run_flops_spectral(main_clusterer_in, D, autotune)
        pre_flops = run_flops_ahc(pre_clusterer_in, pre_clusterer_out, D)
    total_flops = fallback_flops + main_flops + pre_flops
    return [N, D, L, U1, U2, str(autotune),
            fallback_flops, main_flops, pre_flops, total_flops]


def main():
    results = [["N", "D", "L", "U1", "U2", "autotune",
                "fallback clusterer FLOPs", "main clusterer FLOPs",
                "pre-clusterer FLOPs", "total FLOPs"]]
    for autotune in [False, True]:
        for N in [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
            results.append(run_flops(N=N, D=256, L=INF,
                           U1=INF, U2=INF, autotune=autotune))
            results.append(run_flops(N=N, D=256, L=0,  U1=INF,
                           U2=INF, autotune=autotune))
            results.append(run_flops(N=N, D=256, L=50,
                           U1=INF, U2=INF, autotune=autotune))
            results.append(run_flops(N=N, D=256, L=50,
                           U1=300, U2=INF, autotune=autotune))
            results.append(run_flops(N=N, D=256, L=50,
                           U1=100, U2=INF, autotune=autotune))
            results.append(run_flops(N=N, D=256, L=50,
                           U1=300, U2=600, autotune=autotune))
            results.append(run_flops(N=N, D=256, L=50,
                           U1=100, U2=600, autotune=autotune))
            results.append(run_flops(N=N, D=256, L=50,
                           U1=100, U2=300, autotune=autotune))

    output_path = os.path.join(os.path.dirname(__file__), "output.csv")
    with open(output_path, "w") as csvfile:
        writer = csv.writer(csvfile)
        for row in results:
            writer.writerow(row)
    print("Results written to:", output_path)


if __name__ == "__main__":
    main()
