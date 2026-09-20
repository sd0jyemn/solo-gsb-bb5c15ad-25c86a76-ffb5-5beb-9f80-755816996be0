"""自适应双酶切决策树：记忆化极小极大单元测试 + API 端到端测试。"""
import random
from itertools import combinations

from fastapi.testclient import TestClient

from app.main import app
from app.decision_tree import (
    build_equivalence_groups,
    genotype_band_signature,
    optimal_decision_tree,
)

client = TestClient(app)


def _enzyme(name, genotype_sites):
    return {"name": name, "genotype_sites": genotype_sites}


def _payload(enzymes, length=1000, min_visible=10, tolerance=0):
    return {
        "plasmid_length": length,
        "min_visible_fragment": min_visible,
        "band_resolution_tolerance": tolerance,
        "enzymes": enzymes,
    }


# 深度 2、含不可区分组的标准场景（examples/decision-tree-request.json 同构）：
#   A+B: {g1,g2,g4,g5} vs {g3}（2 块）
#   A+C: {g1} vs {g2,g3,g5} vs {g4}（3 块）
#   B+C: {g1} vs {g2,g4,g5} vs {g3}（3 块）
# g2/g5 在全部酶对上签名相同 -> 不可区分；4 个候选类最坏需要 2 次实验。
STANDARD = _payload([
    _enzyme("A", {"g1": [0, 100], "g2": [0, 100], "g3": [0, 100],
                  "g4": [0, 300], "g5": [0, 100]}),
    _enzyme("B", {"g1": [400], "g2": [400], "g3": [200], "g4": [400], "g5": [400]}),
    _enzyme("C", {"g1": [700], "g2": [550], "g3": [550], "g4": [550], "g5": [550]}),
])


# ------------------------------------------------- 规范化签名（复用圆环/聚带规则）

def test_signature_reuses_circular_and_clustering_rules():
    # 切点 0 < 100 < 208（L=1000）-> 片段 100、108、792（末段跨原点）
    # 容差 10：100 与 108 单连接成带（均值 104），792 独立成带
    sig = genotype_band_signature([0], [100, 208], "A", "B", 1000, 1, 10)
    assert sig == (104.0, 792.0)


def test_signature_visible_threshold_filters_short_fragments():
    # 片段 100、108、792；可见阈值 200 时只剩 792
    sig = genotype_band_signature([0], [100, 208], "A", "B", 1000, 200, 10)
    assert sig == (792.0,)


def test_signature_empty_when_fewer_than_two_merged_cuts():
    # 两酶同位点（合并后 1 个切点）无法完成圆环双酶切 -> 空签名
    assert genotype_band_signature([100], [100], "A", "B", 1000, 1, 0) == ()
    # 单酶单切点 -> 空签名
    assert genotype_band_signature([100], [], "A", "B", 1000, 1, 0) == ()


def test_equivalence_groups_merge_identical_signature_vectors():
    pairs = ["A+B", "A+C"]
    signatures = {
        ("A+B", "x"): (1.0, 2.0), ("A+C", "x"): (3.0,),
        ("A+B", "y"): (1.0, 2.0), ("A+C", "y"): (3.0,),
        ("A+B", "z"): (9.0,), ("A+C", "z"): (3.0,),
    }
    groups = build_equivalence_groups(["z", "y", "x"], pairs, signatures)
    # 全向量相同才合并；组内字典序、组间按首名字典序
    assert groups == [["x", "y"], ["z"]]


# ------------------------------------------------- 极小极大纯函数

def test_optimal_tree_single_or_unseparable_candidates():
    leaf = optimal_decision_tree([0], [("A+B", {(1.0,): frozenset({0})})])
    assert leaf.height == 0 and leaf.pair_key is None and leaf.internal_count == 0

    none_split = optimal_decision_tree(
        [0, 1], [("A+B", {(1.0,): frozenset({0, 1})})],
    )
    assert none_split.height == 0 and none_split.pair_key is None


def test_optimal_tree_single_experiment():
    root = optimal_decision_tree(
        [0, 1, 2],
        [("A+B", {(1.0,): frozenset({0, 1}), (2.0,): frozenset({2})})],
    )
    assert root.height == 1 and root.internal_count == 1 and root.pair_key == "A+B"


def test_optimal_tree_minimax_depth_two():
    # 两个酶对各只能二分，且划分交错 -> 最坏需要 2 次实验
    root = optimal_decision_tree([0, 1, 2], [
        ("A+B", {(1.0,): frozenset({0, 1}), (2.0,): frozenset({2})}),
        ("A+C", {(1.0,): frozenset({0}), (2.0,): frozenset({1, 2})}),
    ])
    assert root.height == 2
    assert root.internal_count == 2  # 根 + 1 个内部子节点


def test_optimal_tree_ignores_non_splitting_pairs():
    root = optimal_decision_tree([0, 1], [
        ("A+B", {(1.0,): frozenset({0, 1})}),       # 根上无法拆分 -> 忽略
        ("A+C", {(1.0,): frozenset({0}), (2.0,): frozenset({1})}),
    ])
    assert root.pair_key == "A+C" and root.height == 1


def test_optimal_tree_dedupes_identical_partitions_by_key():
    blocks = {(1.0,): frozenset({0}), (2.0,): frozenset({1})}
    root = optimal_decision_tree([0, 1], [
        ("Z+Z", blocks), ("A+B", blocks),
    ])
    # 同构酶对只保留 pair_key 字典序最小者
    assert root.pair_key == "A+B"


def test_optimal_tree_tie_break_prefers_fewer_internal_nodes():
    # A+A：{0}|{1,2,3}，剩余 3 类可被 Q 一次三分开 -> 内部节点 2
    # B+B：{0,1}|{2,3}，两块都需再做一次实验（A+A / Q）-> 内部节点 3
    # Q：   {1}|{2}|{0,3} -> {0,3} 再做 A+A -> 内部节点 2、键序更大
    root = optimal_decision_tree([0, 1, 2, 3], [
        ("B+B", {(1.0,): frozenset({0, 1}), (2.0,): frozenset({2, 3})}),
        ("Q+Q", {(1.0,): frozenset({1}), (2.0,): frozenset({2}),
                 (3.0,): frozenset({0, 3})}),
        ("A+A", {(1.0,): frozenset({0}), (2.0,): frozenset({1, 2, 3})}),
    ])
    assert root.height == 2
    assert root.pair_key == "A+A"
    assert root.internal_count == 2


def test_optimal_tree_tie_break_lexicographic_pair_key():
    # 两酶对同深度同内部节点数（各二分一次拆完两个类），取键序小者
    root = optimal_decision_tree([0, 1], [
        ("B+C", {(1.0,): frozenset({0}), (2.0,): frozenset({1})}),
        ("A+B", {(9.0,): frozenset({0}), (8.0,): frozenset({1})}),
    ])
    assert (root.height, root.pair_key) == (1, "A+B")


def test_optimal_tree_matches_brute_force():
    """与暴力枚举全部自适应策略交叉验证：最坏深度 -> 内部节点数 -> 根键。"""
    def brute_force(cands, tests):
        best_key: dict = {}
        sigs_by: dict = {}
        for key, blocks in tests:
            sigs = tuple(sorted(blocks, key=lambda s: tuple(sorted(blocks[s]))))
            blk = tuple(blocks[s] for s in sigs)
            if len(blk) <= 1:
                continue
            if blk not in best_key or key < best_key[blk]:
                best_key[blk] = key
                sigs_by[blk] = sigs
        uniq = sorted(
            ((best_key[b], sigs_by[b], b) for b in best_key), key=lambda t: t[0],
        )

        import functools

        @functools.lru_cache(None)
        def solve(members, remaining):
            members = frozenset(members)
            remaining = frozenset(remaining)
            if len(members) <= 1:
                return (0, 0, None)
            best = None
            for key, sigs, blocks in uniq:
                if key not in remaining:
                    continue
                subs = [blocks[i] & members for i in range(len(blocks))]
                if len([s for s in subs if s]) < 2:
                    continue
                ch = [solve(s, remaining - {key}) for s in subs if s]
                h = 1 + max(c[0] for c in ch)
                ic = 1 + sum(c[1] for c in ch)
                cand = (h, ic, key)
                if best is None or cand < best:
                    best = cand
            return best if best is not None else (0, 0, None)

        return solve(frozenset(cands), frozenset(k for k, _, _ in uniq))

    rng = random.Random(20260920)
    for _ in range(400):
        n = rng.randint(1, 5)
        cands = list(range(n))
        tests = []
        for ti in range(rng.randint(0, 6)):
            blocks = {}
            for c in cands:
                blocks.setdefault((float(rng.randint(0, 2)),), set()).add(c)
            tests.append((f"K{ti:02d}", {s: frozenset(v) for s, v in blocks.items()}))
        got = optimal_decision_tree(cands, tests)
        exp = brute_force(cands, tests)
        assert (got.height, got.internal_count, got.pair_key) == exp


# ---------------------------------------------------------------- API 基础

def _leaves(node):
    if node["type"] == "leaf":
        return [node]
    out = []
    for branch in node["branches"]:
        out.extend(_leaves(branch["child"]))
    return out


def test_decision_tree_standard_scenario():
    res = client.post("/api/v1/digest/decision-tree", json=STANDARD)
    assert res.status_code == 200
    body = res.json()

    assert body["genotype_count"] == 5
    assert body["total_enzyme_pairs"] == 3
    assert body["worst_case_depth"] == 2
    assert body["internal_node_count"] == 2
    assert body["leaf_count"] == 3
    assert body["indistinguishable_groups"] == [["g2", "g5"]]

    root = body["tree"]
    assert root["type"] == "internal" and root["depth"] == 0
    assert root["pair_key"] == "A+B" and root["enzyme_pair"] == ["A", "B"]
    assert root["candidate_count"] == 4

    leaves = _leaves(root)
    assert sorted(tuple(leaf["genotypes"]) for leaf in leaves) == [
        ("g1",), ("g2", "g5"), ("g3",), ("g4",),
    ]
    assert max(leaf["depth"] for leaf in leaves) == 2

    # 每个实验节点的分支签名互不相同且升序（空签名场景另测）
    def check(node):
        if node["type"] == "internal":
            sigs = [tuple(b["signature"]) for b in node["branches"]]
            assert sigs == sorted(sigs)
            assert len(set(sigs)) == len(sigs)
            for branch in node["branches"]:
                check(branch["child"])
    check(root)

    # 逐酶对签名：键序、空切点不足名单齐全，可由叶子签名反推分组
    assert [p["pair_key"] for p in body["pair_signatures"]] == ["A+B", "A+C", "B+C"]
    sig_map = {p["pair_key"]: p["signatures"] for p in body["pair_signatures"]}
    assert sig_map["A+B"]["g2"] == sig_map["A+B"]["g5"]
    assert sig_map["A+C"]["g2"] == sig_map["A+C"]["g5"]
    assert sig_map["B+C"]["g2"] == sig_map["B+C"]["g5"]
    assert all(p["insufficient_cut_genotypes"] == [] for p in body["pair_signatures"])
    assert body["reason"]


def test_decision_tree_root_branch_paths_classify_every_genotype():
    body = client.post("/api/v1/digest/decision-tree", json=STANDARD).json()
    classified = [g for leaf in _leaves(body["tree"]) for g in leaf["genotypes"]]
    assert sorted(classified) == ["g1", "g2", "g3", "g4", "g5"]


def test_decision_tree_depth_one_when_a_pair_distinguishes_all():
    # A+C 在 4 个不同基因型上给 4 个不同签名 -> 一次实验即可
    payload = _payload([
        _enzyme("A", {"g1": [0, 100], "g2": [0, 100], "g3": [0, 100], "g4": [0, 300]}),
        _enzyme("B", {"g1": [400], "g2": [400], "g3": [200], "g4": [400]}),
        _enzyme("C", {"g1": [700], "g2": [550], "g3": [650], "g4": [550]}),
    ])
    body = client.post("/api/v1/digest/decision-tree", json=payload).json()
    assert body["indistinguishable_groups"] == []
    assert body["worst_case_depth"] == 1
    assert body["internal_node_count"] == 1
    assert body["tree"]["pair_key"] == "A+C"
    assert len(body["tree"]["branches"]) == 4


def test_decision_tree_all_indistinguishable_is_single_leaf():
    payload = _payload([
        _enzyme("EcoRI", {"v1": [100, 600], "v2": [100, 600]}),
        _enzyme("BamHI", {"v1": [300], "v2": [300]}),
    ])
    body = client.post("/api/v1/digest/decision-tree", json=payload).json()
    assert body["worst_case_depth"] == 0
    assert body["internal_node_count"] == 0
    assert body["leaf_count"] == 1
    assert body["indistinguishable_groups"] == [["v1", "v2"]]
    leaf = body["tree"]
    assert leaf["type"] == "leaf" and leaf["depth"] == 0
    assert leaf["genotypes"] == ["v1", "v2"]
    assert leaf["branches"] == [] and leaf["pair_key"] is None


def test_decision_tree_single_genotype_is_trivial_leaf():
    payload = _payload([
        _enzyme("EcoRI", {"only": [100, 600]}),
        _enzyme("BamHI", {"only": [300]}),
    ])
    body = client.post("/api/v1/digest/decision-tree", json=payload).json()
    assert body["genotype_count"] == 1
    assert body["worst_case_depth"] == 0
    assert body["tree"]["type"] == "leaf"
    assert body["tree"]["genotypes"] == ["only"]
    assert body["indistinguishable_groups"] == []


def test_decision_tree_insufficient_cuts_give_empty_signature_first_branch():
    # gx 在 A、B 上同位点 -> 合并仅 1 个切点，签名为空；gy 正常双切点
    payload = _payload([
        _enzyme("A", {"gx": [100], "gy": [100, 500]}),
        _enzyme("B", {"gx": [100], "gy": [800]}),
    ], length=1000, min_visible=1, tolerance=0)
    body = client.post("/api/v1/digest/decision-tree", json=payload).json()
    assert body["worst_case_depth"] == 1
    pair_info = body["pair_signatures"][0]
    assert pair_info["signatures"]["gx"] == []
    assert pair_info["insufficient_cut_genotypes"] == ["gx"]
    # 空签名分支按元组升序排在最前
    assert body["tree"]["branches"][0]["signature"] == []
    assert body["tree"]["branches"][0]["child"]["genotypes"] == ["gx"]


def test_decision_tree_non_splitting_pair_ignored():
    # A+B 对全部基因型签名相同；A+C 可拆分 -> 根取 A+C，搜索不受 A+B 干扰
    payload = _payload([
        _enzyme("A", {"g1": [0, 100], "g2": [0, 100]}),
        _enzyme("B", {"g1": [400], "g2": [400]}),
        _enzyme("C", {"g1": [700], "g2": [550]}),
    ])
    body = client.post("/api/v1/digest/decision-tree", json=payload).json()
    assert body["tree"]["pair_key"] == "A+C"
    assert body["worst_case_depth"] == 1


def test_decision_tree_order_independent():
    first = client.post("/api/v1/digest/decision-tree", json=STANDARD).json()

    shuffled = {
        "plasmid_length": STANDARD["plasmid_length"],
        "min_visible_fragment": STANDARD["min_visible_fragment"],
        "band_resolution_tolerance": STANDARD["band_resolution_tolerance"],
        "enzymes": [
            {"name": "C", "genotype_sites": {
                "g5": [550], "g2": [550], "g4": [550], "g1": [700], "g3": [550]}},
            {"name": "A", "genotype_sites": {
                "g4": [0, 300], "g2": [0, 100], "g1": [0, 100], "g5": [0, 100],
                "g3": [0, 100]}},
            {"name": "B", "genotype_sites": {
                "g3": [200], "g1": [400], "g5": [400], "g4": [400], "g2": [400]}},
        ],
    }
    second = client.post("/api/v1/digest/decision-tree", json=shuffled).json()
    assert first == second


def test_decision_tree_tolerance_chains_adjacent_fragments():
    # g1 片段 100/108/792，容差 10 -> 100 与 108 单连接同带；g2 全部差异明显
    payload = _payload([
        _enzyme("A", {"g1": [0], "g2": [0]}),
        _enzyme("B", {"g1": [100, 208], "g2": [400, 700]}),
    ], length=1000, min_visible=1, tolerance=10)
    body = client.post("/api/v1/digest/decision-tree", json=payload).json()
    sig = body["pair_signatures"][0]["signatures"]
    assert sig["g1"] == [104.0, 792.0]


# ---------------------------------------------------------------- 422 校验

def _expect_422(payload, code_part=None):
    res = client.post("/api/v1/digest/decision-tree", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert body["errors"]
    if code_part:
        assert any(code_part in e["type"] for e in body["errors"])
    return body


def test_422_site_out_of_range():
    _expect_422(_payload([
        _enzyme("A", {"g1": [100, 1000]}),   # 1000 越界（合法 0..999）
        _enzyme("B", {"g1": [300]}),
    ]), "site_out_of_range")


def test_422_negative_site():
    _expect_422(_payload([
        _enzyme("A", {"g1": [-1]}),
        _enzyme("B", {"g1": [300]}),
    ]), "site_out_of_range")


def test_422_duplicate_site():
    _expect_422(_payload([
        _enzyme("A", {"g1": [100, 100, 600]}),
        _enzyme("B", {"g1": [300]}),
    ]), "duplicate_site")


def test_422_duplicate_enzyme_name():
    _expect_422(_payload([
        _enzyme("A", {"g1": [100]}),
        _enzyme("A", {"g1": [300]}),
    ]), "duplicate_enzyme_name")


def test_422_inconsistent_genotype_names():
    _expect_422(_payload([
        _enzyme("A", {"g1": [100]}),
        _enzyme("B", {"g2": [300]}),
    ]), "inconsistent_genotype_names")


def test_422_blank_genotype_name():
    _expect_422(_payload([
        _enzyme("A", {"": [100]}),
        _enzyme("B", {"": [300]}),
    ]), "invalid_genotype_name")


def test_422_no_genotypes():
    _expect_422(_payload([
        _enzyme("A", {}),
        _enzyme("B", {}),
    ]), "no_genotypes")


def test_422_too_many_genotypes(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_DECISION_TREE_GENOTYPES", 1)
    _expect_422(_payload([
        _enzyme("A", {"g1": [100], "g2": [200]}),
        _enzyme("B", {"g1": [300], "g2": [400]}),
    ]), "too_many_decision_tree_genotypes")


def test_422_too_many_enzymes(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_DECISION_TREE_ENZYMES", 2)
    _expect_422(_payload([
        _enzyme("A", {"g1": [100]}),
        _enzyme("B", {"g1": [300]}),
        _enzyme("C", {"g1": [500]}),
    ]), "too_many_decision_tree_enzymes")


def test_422_independent_enzyme_limit_does_not_change_panel_limit(monkeypatch):
    # 独立上限：调小 decision-tree 酶上限不影响 MAX_ENZYMES 本身的配置值
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_DECISION_TREE_ENZYMES", 2)
    assert settings.MAX_ENZYMES == 64


def test_422_min_visible_not_smaller_than_plasmid():
    _expect_422(_payload(
        [_enzyme("A", {"g1": [100]}), _enzyme("B", {"g1": [300]})],
        min_visible=1000,
    ), "min_visible_too_large")


def test_422_basic_field_constraints():
    # 质粒长度 <= 0、容差为负、酶不足 2 种、genotype_sites 类型错误
    _expect_422(_payload(
        [_enzyme("A", {"g1": [100]}), _enzyme("B", {"g1": [300]})],
        length=0,
    ))
    _expect_422(_payload(
        [_enzyme("A", {"g1": [100]}), _enzyme("B", {"g1": [300]})],
        tolerance=-1,
    ))
    _expect_422(_payload([_enzyme("A", {"g1": [100]})]))
    _expect_422(_payload([
        {"name": "A", "genotype_sites": {"g1": [[100]]}},
        _enzyme("B", {"g1": [300]}),
    ]))
    # 缺必填字段
    _expect_422({"enzymes": [_enzyme("A", {"g1": [100]}), _enzyme("B", {"g1": [300]})]})
