import pandas as pd
import numpy as np
from scipy.stats import rankdata


def get_unique_concepts(concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_category_assignments = concepts.sum(axis=1)
    unique_memberships = np.where(
        num_category_assignments > 1.0, 0.0, num_category_assignments
    ).astype(bool)
    singletons = concepts.iloc[unique_memberships, :]
    non_singletons = concepts.iloc[~unique_memberships, :]
    return singletons, non_singletons


def sort_concepts(concepts: pd.DataFrame) -> np.ndarray:
    return np.hstack(
        [concepts[concepts.loc[:, concept] == 1.0].index for concept in concepts.keys()]
    )


def filter_rsm_by_things_concepts(
    rsm_human: np.ndarray, rsm_dnn: np.ndarray, concepts: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Sort human and DNN by their assigned concept concepts, so that
    objects belonging to the same concept are grouped together in the RSM"""

    singletons, non_singletons = get_unique_concepts(concepts)
    singletons = sort_concepts(singletons)
    non_singletons = np.random.permutation(non_singletons.index)
    sorted_items = np.hstack((singletons, non_singletons))

    rsm_human = rsm_human[sorted_items, :]
    rsm_human = rsm_human[:, sorted_items]

    rsm_dnn = rsm_dnn[sorted_items, :]
    rsm_dnn = rsm_dnn[:, sorted_items]
    rsm_human = rankdata(rsm_human).reshape(rsm_human.shape)
    rsm_dnn = rankdata(rsm_dnn).reshape(rsm_dnn.shape)
    return rsm_human, rsm_dnn
