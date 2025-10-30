# cython: language_level=3
"""Splitting algorithms in the construction of a tree.

This module contains the main splitting algorithms for constructing a tree.
Splitting is concerned with finding the optimal partition of the data into
two groups. The impurity of the groups is minimized, and the impurity is measured
by some criterion, which is typically the Gini impurity or the entropy. Criterion
are implemented in the ``_criterion`` module.

Splitting evaluates a subset of features (defined by `max_features` also
known as mtry in the literature). The module supports two primary types
of splitting strategies:

- Best Split: A greedy approach to find the optimal split. This method
  ensures that the best possible split is chosen by examining various
  thresholds for each candidate feature.
- Random Split: A stochastic approach that selects a split randomly
  from a subset of the best splits. This method is faster but does
  not guarantee the optimal split.
"""
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.string cimport memcpy

from sklearn.tree._splitter cimport Splitter
from sklearn.tree._splitter cimport SplitRecord
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._tree cimport ParentInfo
from sklearn.tree._partitioner cimport DensePartitioner,SparsePartitioner

from sklearn.utils._typedefs cimport (
    float32_t, float64_t, int8_t, int32_t, intp_t, uint8_t, uint32_t
)
from sklearn.tree._partitioner cimport (
    FEATURE_THRESHOLD, DensePartitioner, SparsePartitioner,
    shift_missing_values_to_left_if_required
)
#from _QuantilePartitioner import QuantileDensePartitioner
from sklearn.tree._utils cimport RAND_R_MAX, rand_int, rand_uniform

import numpy as np
cimport numpy as np

cdef float64_t INFINITY = np.inf

cdef inline void _init_split(SplitRecord* self, intp_t start_pos) noexcept nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.missing_go_to_left = False
    self.n_missing = 0

# Introduce a fused-class to make it possible to share the split implementation
# between the dense and sparse cases in the node_split_best and node_split_random
# functions. The alternative would have been to use inheritance-based polymorphism
# but it would have resulted in a ~10% overall tree fitting performance
# degradation caused by the overhead frequent virtual method lookups.
ctypedef fused Partitioner:
    DensePartitioner
    SparsePartitioner

cdef inline int node_split_best(
    QuantileBestSplitter splitter,
    Partitioner partitioner,
    Criterion criterion,
    SplitRecord* split,
    ParentInfo* parent_record,
    ) except -1 nogil:
        """Find the best split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef const int8_t[:] monotonic_cst = splitter.monotonic_cst
        cdef bint with_monotonic_cst = splitter.with_monotonic_cst

        # Find the best split
        cdef intp_t start = splitter.start
        cdef intp_t end = splitter.end
        cdef intp_t end_non_missing
        cdef intp_t n_missing = 0
        cdef bint has_missing = 0
        cdef intp_t n_searches
        cdef intp_t n_left, n_right
        cdef bint missing_go_to_left

        cdef intp_t[::1] samples = splitter.samples
        cdef intp_t[::1] features = splitter.features
        cdef intp_t[::1] constant_features = splitter.constant_features
        cdef intp_t n_features = splitter.n_features

        cdef float32_t[::1] feature_values = splitter.feature_values
        cdef intp_t max_features = splitter.max_features
        cdef intp_t min_samples_leaf = splitter.min_samples_leaf
        cdef float64_t min_weight_leaf = splitter.min_weight_leaf
        cdef uint32_t* random_state = &splitter.rand_r_state

        cdef SplitRecord best_split, current_split
        cdef float64_t current_proxy_improvement = -INFINITY
        cdef float64_t best_proxy_improvement = -INFINITY

        cdef float64_t impurity = parent_record.impurity
        cdef float64_t lower_bound = parent_record.lower_bound
        cdef float64_t upper_bound = parent_record.upper_bound

        cdef intp_t f_i = n_features
        cdef intp_t f_j
        cdef intp_t p
        cdef intp_t p_prev

        cdef intp_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef intp_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef intp_t n_drawn_constants = 0
        cdef intp_t n_known_constants = parent_record.n_constant_features
        # n_total_constants = n_known_constants + n_found_constants
        cdef intp_t n_total_constants = n_known_constants

        _init_split(&best_split, end)

        partitioner.init_node_split(start, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                # At least one drawn features must be non constant
                n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                        random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

                n_drawn_constants += 1
                continue

            # f_j in the interval [n_known_constants, f_i - n_found_constants[
            f_j += n_found_constants
            # f_j in the interval [n_total_constants, f_i[
            current_split.feature = features[f_j]
            partitioner.sort_samples_and_feature_values(current_split.feature)
            n_missing = partitioner.n_missing
            end_non_missing = end - n_missing

            if (
                # All values for this feature are missing, or
                end_non_missing == start or
                # This feature is considered constant (max - min <= FEATURE_THRESHOLD)
                feature_values[end_non_missing - 1] <= feature_values[start] + FEATURE_THRESHOLD
            ):
                # We consider this feature constant in this case.
                # Since finding a split among constant feature is not valuable,
                # we do not consider this feature for splitting.
                features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                n_found_constants += 1
                n_total_constants += 1
                continue

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            has_missing = n_missing != 0
            criterion.init_missing(n_missing)  # initialize even when n_missing == 0

            # Evaluate all splits

            # If there are missing values, then we search twice for the most optimal split.
            # The first search will have all the missing values going to the right node.
            # The second search will have all the missing values going to the left node.
            # If there are no missing values, then we search only once for the most
            # optimal split.
            n_searches = 2 if has_missing else 1

            for i in range(n_searches):
                missing_go_to_left = i == 1
                criterion.missing_go_to_left = missing_go_to_left
                criterion.reset()

                p = start

                while p < end_non_missing:
                    partitioner.next_p(&p_prev, &p)

                    if p >= end_non_missing:
                        continue

                    if missing_go_to_left:
                        n_left = p - start + n_missing
                        n_right = end_non_missing - p
                    else:
                        n_left = p - start
                        n_right = end_non_missing - p + n_missing

                    # Reject if min_samples_leaf is not guaranteed
                    if n_left < min_samples_leaf or n_right < min_samples_leaf:
                        continue

                    current_split.pos = p
                    criterion.update(current_split.pos)

                    # Reject if monotonicity constraints are not satisfied
                    if (
                        with_monotonic_cst and
                        monotonic_cst[current_split.feature] != 0 and
                        not criterion.check_monotonicity(
                            monotonic_cst[current_split.feature],
                            lower_bound,
                            upper_bound,
                        )
                    ):
                        continue

                    # Reject if min_weight_leaf is not satisfied
                    if ((criterion.weighted_n_left < min_weight_leaf) or
                            (criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current_proxy_improvement = criterion.proxy_impurity_improvement()

                    if current_proxy_improvement > best_proxy_improvement:
                        best_proxy_improvement = current_proxy_improvement
                        # sum of halves is used to avoid infinite value
                        current_split.threshold = (
                            feature_values[p_prev] # / 2.0 + feature_values[p] / 2.0
                        )

                        if (
                            current_split.threshold == feature_values[p] or
                            current_split.threshold == INFINITY or
                            current_split.threshold == -INFINITY
                        ):
                            current_split.threshold = feature_values[p_prev]

                        current_split.n_missing = n_missing

                        # if there are no missing values in the training data, during
                        # test time, we send missing values to the branch that contains
                        # the most samples during training time.
                        if n_missing == 0:
                            current_split.missing_go_to_left = n_left > n_right
                        else:
                            current_split.missing_go_to_left = missing_go_to_left

                        best_split = current_split  # copy

            # Evaluate when there are missing values and all missing values goes
            # to the right node and non-missing values goes to the left node.
            if has_missing:
                n_left, n_right = end - start - n_missing, n_missing
                p = end - n_missing
                missing_go_to_left = 0

                if not (n_left < min_samples_leaf or n_right < min_samples_leaf):
                    criterion.missing_go_to_left = missing_go_to_left
                    criterion.update(p)

                    if not ((criterion.weighted_n_left < min_weight_leaf) or
                            (criterion.weighted_n_right < min_weight_leaf)):
                        current_proxy_improvement = criterion.proxy_impurity_improvement()

                        if current_proxy_improvement > best_proxy_improvement:
                            best_proxy_improvement = current_proxy_improvement
                            current_split.threshold = INFINITY
                            current_split.missing_go_to_left = missing_go_to_left
                            current_split.n_missing = n_missing
                            current_split.pos = p
                            best_split = current_split

        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        if best_split.pos < end:
            partitioner.partition_samples_final(
                best_split.pos,
                best_split.threshold,
                best_split.feature,
                best_split.n_missing
            )
            criterion.init_missing(best_split.n_missing)
            criterion.missing_go_to_left = best_split.missing_go_to_left

            criterion.reset()
            criterion.update(best_split.pos)
            criterion.children_impurity(
                &best_split.impurity_left, &best_split.impurity_right
            )
            best_split.improvement = criterion.impurity_improvement(
                impurity,
                best_split.impurity_left,
                best_split.impurity_right
            )

            shift_missing_values_to_left_if_required(&best_split, samples, end)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(&features[0], &constant_features[0], sizeof(intp_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(&constant_features[n_known_constants],
            &features[n_known_constants],
            sizeof(intp_t) * n_found_constants)

        # Return values
        parent_record.n_constant_features = n_total_constants
        split[0] = best_split
        return 0

cdef class QuantileBestSplitter(Splitter):
    """Splitter for finding the best split on dense data."""
    cdef DensePartitioner partitioner
    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )

    cdef int node_split(
            self,
            ParentInfo* parent_record,
            SplitRecord* split,
    ) except -1 nogil:
        return node_split_best(
            self,
            self.partitioner,
            self.criterion,
            split,
            parent_record,
        )
