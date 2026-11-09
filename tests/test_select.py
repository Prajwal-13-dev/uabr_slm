from uabr.retrieval.select import select_non_overlapping


def test_skip_overlapping_same_article():
    art = {1: 0, 2: 0, 3: 1, 4: 0, 5: 2}
    st = {1: 0, 2: 30, 3: 0, 4: 60, 5: 0}
    chosen, skipped = select_non_overlapping([1, 2, 3, 4, 5], art, st, k=3)
    assert chosen == [1, 3, 4] and skipped == 1     # 2 overlaps 1; 4 starts at 60 -> no overlap
    assert select_non_overlapping([1, 2, -1], art, st, k=3) == ([1], 1)
