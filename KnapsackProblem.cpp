
#include <vector>
#include <iostream>
#include <tuple>
#include <algorithm>
#include <chrono>
#include <random>
#include <numeric>
#include <immintrin.h>
#include <thread>

using namespace std;

struct ThreadResult {
    int bestValue;
    int bestMask;
};

tuple<int, vector<int>> knapsackBruteForceMultiThreadedAVX512(const int W, const vector<int>& weights, const vector<int>& values) {
    int N = weights.size();
    long long num_combinations = (long long)1 << N;
    int bestValue = 0;
    int bestMask = 0; // To store the bitmask of the best subset

    // Convert weights and values to aligned arrays for AVX
    alignas(64) vector<int> weight_array(weights);
    alignas(64) vector<int> value_array(values);

    // Determine the number of threads
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;

    // To store the results from each thread
    vector<ThreadResult> thread_results(num_threads);

    // Create and launch threads
    vector<thread> threads(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        threads[t] = thread([&, t]() {
            long long start = t * num_combinations / num_threads;
            long long end = (t + 1) * num_combinations / num_threads;
            int localBestValue = 0;
            int localBestMask = 0;

            long long adjusted_end = end - (end % 16);
            if (adjusted_end < start) adjusted_end = start;

            // Iterate over all subsets assigned to this thread in chunks of 16 (for AVX-512)
            for (long long i = start; i < adjusted_end; i += 16) {
                // Load current batch of 16 bitmasks
                __m512i masks = _mm512_set_epi32(
                    (int)(i + 15), (int)(i + 14), (int)(i + 13), (int)(i + 12), (int)(i + 11), (int)(i + 10), (int)(i + 9), (int)(i + 8),
                    (int)(i + 7), (int)(i + 6), (int)(i + 5), (int)(i + 4), (int)(i + 3), (int)(i + 2), (int)(i + 1), (int)i
                );

                // Initialize accumulators for weights and values
                __m512i currentWeights = _mm512_setzero_si512();
                __m512i currentValues = _mm512_setzero_si512();

                for (int j = 0; j < N; j++) {
                    // Check if the j-th item is included in each subset
                    __m512i itemMask = _mm512_set1_epi32(1 << j);
                    __m512i isIncluded = _mm512_and_si512(masks, itemMask);

                    // Create a mask for inclusion
                    __mmask16 inclusionMask = _mm512_test_epi32_mask(isIncluded, itemMask);

                    // Add weights and values for included items
                    __m512i weightVec = _mm512_set1_epi32(weight_array[j]);
                    __m512i valueVec = _mm512_set1_epi32(value_array[j]);

                    currentWeights = _mm512_mask_add_epi32(currentWeights, inclusionMask, currentWeights, weightVec);
                    currentValues = _mm512_mask_add_epi32(currentValues, inclusionMask, currentValues, valueVec);
                }

                // Check if weight is within the capacity
                __m512i capacityVec = _mm512_set1_epi32(W);
                __mmask16 validMask = _mm512_cmp_epi32_mask(currentWeights, capacityVec, _MM_CMPINT_LE);

                // Find maximum value within valid subsets
                __m512i validValues = _mm512_maskz_mov_epi32(validMask, currentValues);

                // Store validValues into an array for extraction
                alignas(64) int validValuesArray[16];
                _mm512_store_epi32(validValuesArray, validValues);

                // Find the local maximum value and its offset
                for (int offset = 0; offset < 16; offset++) {
                    if (validValuesArray[offset] > localBestValue) {
                        localBestValue = validValuesArray[offset];
                        localBestMask = (int)(i + offset);
                    }
                }
            }

            // Process remaining masks
            for (long long i = adjusted_end; i < end; ++i) {
                int currentWeight = 0;
                int currentValue = 0;
                for (int j = 0; j < N; j++) {
                    if (i & (1 << j)) {
                        currentWeight += weight_array[j];
                        currentValue += value_array[j];
                    }
                }
                if (currentWeight <= W && currentValue > localBestValue) {
                    localBestValue = currentValue;
                    localBestMask = (int)i;
                }
            }

            // Store the result
            thread_results[t].bestValue = localBestValue;
            thread_results[t].bestMask = localBestMask;
            });
    }

    // Wait for all threads to finish
    for (size_t t = 0; t < num_threads; ++t) {
        threads[t].join();
    }

    // Combine results from all threads
    for (size_t t = 0; t < num_threads; ++t) {
        if (thread_results[t].bestValue > bestValue) {
            bestValue = thread_results[t].bestValue;
            bestMask = thread_results[t].bestMask;
        }
    }

    // Reconstruct the best subset of weights from the bestMask
    vector<int> bestWeights;
    for (int j = 0; j < N; j++) {
        if (bestMask & (1 << j)) {
            bestWeights.push_back(weights[j]);
        }
    }

    return make_tuple(bestValue, bestWeights);
}

tuple<int, vector<int>> knapsackBruteForceAVX512(const int W, const vector<int>& weights, const vector<int>& values) {
    size_t N = weights.size();
    long long num_combinations = (long long)1 << N;
    int bestValue = 0;
    int bestMask = 0; // To store the bitmask of the best subset

    // Convert weights and values to aligned arrays for AVX
    alignas(64) vector<int> weight_array(weights);
    alignas(64) vector<int> value_array(values);

    // Iterate over all subsets in chunks of 16 (for AVX-512)
    for (long long i = 0; i < num_combinations; i += 16) {
        // Load current batch of 16 bitmasks
        __m512i masks = _mm512_set_epi32(
            i + 15, i + 14, i + 13, i + 12, i + 11, i + 10, i + 9, i + 8,
            i + 7, i + 6, i + 5, i + 4, i + 3, i + 2, i + 1, i
        );

        // Initialize accumulators for weights and values
        __m512i currentWeights = _mm512_setzero_si512();
        __m512i currentValues = _mm512_setzero_si512();

        for (int j = 0; j < N; j++) {
            // Check if the j-th item is included in each subset
            __m512i itemMask = _mm512_set1_epi32(1 << j);
            __m512i isIncluded = _mm512_and_si512(masks, itemMask);

            // Create a mask for inclusion
            __mmask16 inclusionMask = _mm512_test_epi32_mask(isIncluded, itemMask);

            // Add weights and values for included items
            __m512i weightVec = _mm512_set1_epi32(weight_array[j]);
            __m512i valueVec = _mm512_set1_epi32(value_array[j]);

            currentWeights = _mm512_mask_add_epi32(currentWeights, inclusionMask, currentWeights, weightVec);
            currentValues = _mm512_mask_add_epi32(currentValues, inclusionMask, currentValues, valueVec);
        }

        // Check if weight is within the capacity
        __m512i capacityVec = _mm512_set1_epi32(W);
        __mmask16 validMask = _mm512_cmp_epi32_mask(currentWeights, capacityVec, _MM_CMPINT_LE);

        // Find maximum value within valid subsets
        __m512i validValues = _mm512_maskz_mov_epi32(validMask, currentValues);

        // Store validValues into an array for extraction
        alignas(64) int validValuesArray[16];
        _mm512_store_epi32(validValuesArray, validValues);

        // Find the local maximum value and its offset
        for (int offset = 0; offset < 16; offset++) {
            if (validValuesArray[offset] > bestValue) {
                bestValue = validValuesArray[offset];
                bestMask = i + offset; // Store the bitmask of the best subset
            }
        }
    }

    // Reconstruct the best subset of weights from the bestMask
    vector<int> bestWeights;
    for (int j = 0; j < N; j++) {
        if (bestMask & (1 << j)) {
            bestWeights.push_back(weights[j]);
        }
    }

    return make_tuple(bestValue, bestWeights);
}

tuple<int, vector<int>> knapsackBruteForceAVX512WithPruning(const int W, const vector<int>& weights, const vector<int>& values) {
    size_t N = weights.size();
    long long num_combinations = (long long)1 << N;
    int bestValue = 0;
    int bestMask = 0; // To store the bitmask of the best subset

    // Convert weights and values to aligned arrays for AVX
    alignas(64) vector<int> weight_array(weights);
    alignas(64) vector<int> value_array(values);

    // Precompute cumulative weights and values for all items
    int totalWeight = accumulate(weights.begin(), weights.end(), 0);
    int totalValue = accumulate(values.begin(), values.end(), 0);

    // If total weight is within capacity, return total value
    if (totalWeight <= W) {
        return make_tuple(totalValue, weights);
    }

    // Variables to hold the maximum possible value from remaining items
    vector<int> maxValues(N + 1, 0);
    for (int i = N - 1; i >= 0; --i) {
        maxValues[i] = maxValues[i + 1] + values[i];
    }

    // Iterate over all subsets in chunks of 16 (for AVX-512)
    for (long long i = 0; i < num_combinations; i += 16) {
        // Load current batch of 16 bitmasks
        __m512i masks = _mm512_set_epi32(
            (int)(i + 15), (int)(i + 14), (int)(i + 13), (int)(i + 12),
            (int)(i + 11), (int)(i + 10), (int)(i + 9), (int)(i + 8),
            (int)(i + 7), (int)(i + 6), (int)(i + 5), (int)(i + 4),
            (int)(i + 3), (int)(i + 2), (int)(i + 1), (int)i
        );

        // Initialize accumulators for weights and values
        __m512i currentWeights = _mm512_setzero_si512();
        __m512i currentValues = _mm512_setzero_si512();

        // Flag to check if all subsets in the batch can be pruned
        bool allPruned = true;

        for (int j = 0; j < N; j++) {
            // Check if the j-th item is included in each subset
            __m512i itemMask = _mm512_set1_epi32(1 << j);
            __m512i isIncluded = _mm512_and_si512(masks, itemMask);

            // Create a mask for inclusion
            __mmask16 inclusionMask = _mm512_test_epi32_mask(isIncluded, itemMask);

            // Add weights and values for included items
            __m512i weightVec = _mm512_set1_epi32(weight_array[j]);
            __m512i valueVec = _mm512_set1_epi32(value_array[j]);

            currentWeights = _mm512_mask_add_epi32(currentWeights, inclusionMask, currentWeights, weightVec);
            currentValues = _mm512_mask_add_epi32(currentValues, inclusionMask, currentValues, valueVec);
        }

        // Check if weight is within the capacity
        __m512i capacityVec = _mm512_set1_epi32(W);
        __mmask16 validMask = _mm512_cmp_epi32_mask(currentWeights, capacityVec, _MM_CMPINT_LE);

        // Prune batches where no subsets are valid
        if (validMask == 0) {
            continue; // Skip this batch as all subsets exceed capacity
        }

        // Update the allPruned flag
        allPruned = false;

        // Find maximum value within valid subsets
        __m512i validValues = _mm512_maskz_mov_epi32(validMask, currentValues);

        // Store validValues into an array for extraction
        alignas(64) int validValuesArray[16];
        _mm512_store_epi32(validValuesArray, validValues);

        // Store masks into an array for extraction
        alignas(64) int masksArray[16];
        _mm512_store_epi32(masksArray, masks);

        // Prune subsets that cannot improve the current bestValue
        for (int offset = 0; offset < 16; offset++) {
            if (!(validMask & (1 << offset))) {
                continue; // Skip invalid subsets
            }

            // Estimate the maximum possible value from remaining items
            int subsetMask = masksArray[offset];
            int nextItemIndex = _lzcnt_u32(~subsetMask);
            int potentialValue = validValuesArray[offset] + maxValues[nextItemIndex];

            if (potentialValue <= bestValue) {
                continue; // Prune this subset as it cannot improve the best value
            }

            // Update best value if current valid value is better
            if (validValuesArray[offset] > bestValue) {
                bestValue = validValuesArray[offset];
                bestMask = masksArray[offset]; // Store the bitmask of the best subset
            }
        }

        // Early exit if all subsets in the batch are pruned
        if (allPruned) {
            continue;
        }
    }

    // Reconstruct the best subset of weights from the bestMask
    vector<int> bestWeights;
    for (int j = 0; j < N; j++) {
        if (bestMask & (1 << j)) {
            bestWeights.push_back(weights[j]);
        }
    }

    return make_tuple(bestValue, bestWeights);
}

tuple<int, vector<int>> knapsackBruteForce(const int W, const vector<int>& weights, const vector<int>& values) {
    size_t N = weights.size();
    long long num_combinations = (long long)1 << N;
	vector<int> bestWeights;
	int bestValue = 0;

	for (int i = 0; i < num_combinations; i++) {
		int currentWeight = 0;
		int currentValue = 0;
		vector<int> currentWeights;

		for (int j = 0; j < weights.size(); j++) {
			if (i & (1 << j)) {
				currentWeight += weights[j];
				currentValue += values[j];
				currentWeights.emplace_back(weights[j]);
			}
		}

		if (currentWeight <= W && currentValue > bestValue) {
			bestValue = currentValue;
			bestWeights = currentWeights;
		}
	}

	return make_tuple(bestValue, bestWeights);
}


static int KnapsackBruteForceRecursion(int n, int maxWeight, const vector<int>& weights, const vector<int>& values) {
    if (n == 0 || weights.size() != values.size() || maxWeight == 0) return 0;

    // if nth item weight > maxWeight then skip this item
    if (weights[n - 1] > maxWeight) return KnapsackBruteForceRecursion(n - 1, maxWeight, weights, values);

    // split into two -> include item OR do not include item
    int result = max(values[n - 1] + KnapsackBruteForceRecursion(n - 1, maxWeight - weights[n - 1], weights, values), KnapsackBruteForceRecursion(n - 1, maxWeight, weights, values));

    return result;
}

tuple<int, vector<int>> knapsackBruteForceDynamicProgrammingAVX512(const int W, const vector<int>& weights, const vector<int>& values) {
    int N = weights.size();
    if (N == 0 || W == 0) {
        return make_tuple(0, vector<int>());
    }

    // Initialize DP table
    vector<int> dp(W + 1, 0);

    // Vector length for AVX-512
    const int vec_len = 16; // AVX-512 can process 16 integers at once

    for (int i = 0; i < N; ++i) {
        int wt = weights[i];
        int val = values[i];

        if (wt > W) {
            continue; // Skip items that are too heavy
        }

        // Process dp[] array in reverse to avoid overwriting needed values
        for (int w = W; w >= wt; w -= vec_len) {
            int start = max(w - vec_len + 1, wt);
            int count = w - start + 1;

            // Load previous dp values
            __m512i dp_prev = _mm512_loadu_si512((__m512i*)&dp[start - wt]);

            // Create a vector with the current item's value
            __m512i val_vec = _mm512_set1_epi32(val);

            // Load current dp values
            __m512i dp_curr = _mm512_loadu_si512((__m512i*)&dp[start]);

            // Compute new dp values: dp[w] = max(dp[w], dp[w - wt] + val)
            __m512i dp_new = _mm512_add_epi32(dp_prev, val_vec);
            dp_curr = _mm512_max_epi32(dp_curr, dp_new);

            // Store updated dp values
            _mm512_storeu_si512((__m512i*)&dp[start], dp_curr);
        }
    }

    // The maximum value is dp[W]
    int bestValue = dp[W];

    // Backtracking to find the items included in the knapsack
    vector<int> bestWeights;
    int w = W;
    for (int i = N - 1; i >= 0 && bestValue > 0; --i) {
        int wt = weights[i];
        int val = values[i];

        if (w - wt >= 0 && dp[w - wt] + val == dp[w]) {
            bestWeights.push_back(wt);
            w -= wt;
            bestValue -= val;
        }
    }

    reverse(bestWeights.begin(), bestWeights.end());
    return make_tuple(dp[W], bestWeights);
}

long long runBenchmark(int N) {
    // Generate random data
    mt19937 gen(12345);
    uniform_int_distribution<int> dist(1, 1000);

    vector<int> weights(N);
    vector<int> values(N);
    for (int i = 0; i < N; i++) {
        weights[i] = dist(gen);
        values[i] = dist(gen);
    }

    long long totalWeight = accumulate(weights.begin(), weights.end(), 0LL);
    int W = (int)(totalWeight / 2);

    auto start = chrono::high_resolution_clock::now();
    auto [bestValue, bestWeights] = knapsackBruteForceMultiThreadedAVX512(W, weights, values);
    auto end = chrono::high_resolution_clock::now();

    long long durationMs = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    return durationMs;
}

int main() {

    const long long ONE_HOUR_MS = 3600000;

    int currentN = 20; // Starting N 
    int step = 1;      // Increase step
    int last_feasible_N = currentN;
    long long last_time = 0;

    cout << "Starting benchmarking to find the largest N that can be solved under one hour.\n";

    while (true) {
        cout << "Testing N = " << currentN << " ..." << endl;
        long long timeMs = runBenchmark(currentN);

        cout << "N = " << currentN << " took " << timeMs << " ms" << endl;

        if (timeMs < ONE_HOUR_MS) {
            last_feasible_N = currentN;
            last_time = timeMs;
            currentN += step;
        } else {
            cout << "N = " << currentN << " exceeded one hour." << endl;
            break;
        }
    }

    cout << "Largest N that runs under one hour: " << last_feasible_N
        << " with a runtime of " << last_time << " ms" << endl;

    return 0;

	/*vector<int> weights = {  };
	vector<int> values = {  };
	long long W = 0;

	// Generate random values
	mt19937 gen(12345);
	uniform_int_distribution<int> dist(1, 1000);
	int count = 6000;
	for (int i = 0; i < count; i++) {
		weights.emplace_back(dist(gen));
		values.emplace_back(dist(gen));
	}
	long long totalWeight = accumulate(weights.begin(), weights.end(), 0);
	W = totalWeight / 2;

	cout << "Total combinations: " << ((long long)1 << weights.size()) << endl;
	cout << "Max weight: " << W << endl;

	auto start = chrono::high_resolution_clock::now();
	//auto [bestValue, bestWeights] = knapsackBruteForceMultiThreadedAVX512(W, weights, values);
	//auto [bestValue, bestWeights] = knapsackBruteForceAVX512(W, weights, values);
	//auto [bestValue, bestWeights] = knapsackBruteForceAVX512WithPruning(W, weights, values);
	//auto [bestValue, bestWeights] = knapsackBruteForce(W, weights, values);
	auto [bestValue, bestWeights] = knapsackBruteForceDynamicProgrammingAVX512(W, weights, values);
	//int bestValue = KnapsackBruteForceRecursion(weights.size(), W, weights, values);
	auto end = chrono::high_resolution_clock::now();

	cout << "Best value: " << bestValue << endl;
	/*cout << "Best weights: ";
	for (int w : bestWeights) {
		cout << w << " ";
	}
	cout << endl;

	cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

	return 0;*/
}

// count - avx512 - avx512 multi threaded
// 30 - 3372 -
// 31 - 7267
// 32 - 14904
// 33 - 33228
// 34 - 66975
// 35 - 161096
// 40 -  - 620106
