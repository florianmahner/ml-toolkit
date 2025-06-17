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
    rsm_human: np.ndarray, concepts: pd.DataFrame, rank_sort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Sort human and DNN by their assigned concept concepts, so that
    objects belonging to the same concept are grouped together in the RSM"""

    singletons, non_singletons = get_unique_concepts(concepts)
    singletons = sort_concepts(singletons)
    non_singletons = np.random.permutation(non_singletons.index)
    sorted_items = np.hstack((singletons, non_singletons))

    rsm_human = rsm_human[sorted_items, :]
    rsm_human = rsm_human[:, sorted_items]
    if rank_sort:
        rsm_human = rankdata(rsm_human).reshape(rsm_human.shape)
    return rsm_human


def filter_concepts_by_cls_names(
    concepts: pd.DataFrame, cls_names: np.ndarray
) -> np.ndarray:
    """Filter by cls names."""
    # Create a copy of cls_names to preserve the original order
    modified_cls_names = cls_names.copy()

    # Remove "camera" and "file" from the copy
    modified_cls_names = modified_cls_names[
        (modified_cls_names != "camera") & (modified_cls_names != "file")
    ]

    # Append "camera1" and "file1" to the end of the copy
    modified_cls_names = np.append(modified_cls_names, ["camera1", "file1"])

    # Replace white space with underscore in modified_cls_names
    modified_cls_names = [s.replace(" ", "_") for s in modified_cls_names]

    # Create a dictionary to map modified_cls_names to their indices
    name_to_index = {name: index for index, name in enumerate(modified_cls_names)}

    # Filter concepts and sort them based on the order in modified_cls_names
    filtered_concepts = concepts[concepts["uniqueID"].isin(modified_cls_names)]
    filtered_concepts = filtered_concepts.sort_values(
        by="uniqueID", key=lambda x: x.map(name_to_index)
    )

    indices = filtered_concepts.index
    return indices
