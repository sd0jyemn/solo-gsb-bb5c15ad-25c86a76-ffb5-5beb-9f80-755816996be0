"""核心算法：自适应双酶切决策树（记忆化极小极大，精确最小化最坏实验次数）。

所有函数均为纯函数，不依赖任何外部服务或全局状态。

规范化条带签名
--------------
对每个酶对（酶名按字典序规范化）与每个基因型，复用 :mod:`app.digest` 的
圆环双酶切（``merge_cut_points`` / ``circular_fragments``）与单连接聚带
规则（``cluster_bands``）**单独成带**：合并两酶切点后按圆环相邻切点计算
片段，长度 < 最小可见片段的片段不参与成带；可见片段按长度升序扫描，
相邻长度差 ≤ 分辨容差归入同一条带（链式可传递的单连接聚类）。各条带
代表长度（簇内均值，保留两位小数）按升序组成的元组，即该基因型在该
酶对上的**规范化条带签名**；有效切点不足 2 个（无法完成圆环双酶切）或
可见片段为空时签名为空元组 ``()``。

不可区分组
----------
两个基因型在**全部酶对**上的规范化签名都相同，则无论按什么顺序做哪些
酶对实验都永远无法区分；据此把基因型划分为等价类，决策树只需在等价类
之间识别。

记忆化极小极大
--------------
状态为“候选等价类集合 S + 剩余酶对集合”（沿路径用过的酶对在 S 上签名
恒定，等价于已从候选动作中消失）：

  * ``|S| ≤ 1``：高度 0 的叶子（已唯一确定）；
  * 否则依次尝试在 S 上**能拆分 S**（产生 ≥ 2 个非空签名块）的酶对，按该
    酶对上的签名把 S 分块，对每块递归，子树高度取最大值，
    当前树高为 ``1 + max(子树高)``；
  * 选择树高最小的酶对；同高时**内部节点数**（``1 + Σ 子树内部节点数``）
    更少者优先；再同则 **pair_key 字典序**更小者优先；
  * **不能拆分当前集合的酶对直接忽略**（根上只有一个签名块的酶对甚至不会
    进入搜索）。

状态记忆化以候选集合 ``S`` 为键：在 ``S`` 上签名恒定的酶对在 ``S`` 的
任意子集中也恒定，剔除它们不改变精确解；在 ``S`` 上限制划分（非空块的
候选集合）完全相同的酶对互为孪生，只需保留 pair_key 字典序最小者作为
规范代表。酶对、基因型、签名一律按名称 / 长度排序规范化，
**结果与输入提交顺序无关**。
"""
from dataclasses import dataclass
from itertools import combinations

from app.digest import cluster_bands, digest_sample, pair_key
from app.models import (
    DecisionTreeBranchOut,
    DecisionTreeEnzymeInput,
    DecisionTreeNodeOut,
    DecisionTreePairSignatureOut,
    DecisionTreeRequest,
    DecisionTreeResponse,
)


# --------------------------------------------------------------- 规范化签名

def genotype_band_signature(sites_a: list[int], sites_b: list[int],
                            enzyme_a: str, enzyme_b: str, plasmid_length: int,
                            min_visible: int, tolerance: int) -> tuple[float, ...]:
    """单个基因型在单个酶对上的规范化条带签名。

    复用推荐接口的圆环酶切与单连接聚带规则：把该基因型同时当作两样本
    传入（签名只关心条带长度，不关心样本计数）。切点不足 2 个或无可见
    片段时返回空元组。
    """
    site_map = {
        enzyme_a: {"reference": sorted(sites_a), "variant": sorted(sites_a)},
        enzyme_b: {"reference": sorted(sites_b), "variant": sorted(sites_b)},
    }
    digest = digest_sample(
        site_map, enzyme_a, enzyme_b, "reference", plasmid_length, min_visible,
    )
    if len(digest.cuts) < 2:
        return ()
    bands = cluster_bands({"reference": digest}, tolerance, min_visible)
    return tuple(band.representative_length for band in bands)


def compute_pair_signatures(
    enzymes: list[DecisionTreeEnzymeInput], genotypes: list[str],
    plasmid_length: int, min_visible: int, tolerance: int,
) -> tuple[list[tuple[str, list[str]]], dict[tuple[str, str], tuple[float, ...]], dict[str, list[str]]]:
    """枚举全部酶对，逐基因型计算规范化签名。

    返回 ``(酶对列表, {(pair_key, 基因型): 签名}, {pair_key: 切点不足的基因型})``；
    酶对列表为 ``(pair_key, [酶名, 酶名])``，按 pair_key 字典序排列。
    """
    site_sets = {
        enzyme.name: {g: sorted(enzyme.genotype_sites[g]) for g in genotypes}
        for enzyme in enzymes
    }
    names = sorted(site_sets)

    pair_infos: list[tuple[str, list[str]]] = []
    signatures: dict[tuple[str, str], tuple[float, ...]] = {}
    insufficient: dict[str, list[str]] = {}
    for a, b in combinations(names, 2):
        pair, key = pair_key(a, b)
        pair_infos.append((key, pair))
        bad: list[str] = []
        for genotype in genotypes:
            sig = genotype_band_signature(
                site_sets[a][genotype], site_sets[b][genotype],
                a, b, plasmid_length, min_visible, tolerance,
            )
            signatures[(key, genotype)] = sig
            if not sig:
                bad.append(genotype)
        if bad:
            insufficient[key] = bad
    pair_infos.sort(key=lambda info: info[0])
    return pair_infos, signatures, insufficient


# --------------------------------------------------------------- 不可区分组

def build_equivalence_groups(
    genotypes: list[str], pair_keys: list[str],
    signatures: dict[tuple[str, str], tuple[float, ...]],
) -> list[list[str]]:
    """按“全部酶对上签名相同”合并永远无法区分的基因型。

    返回各组的基因型名列表（组内字典序），各组按组内首个名字字典序排列。
    """
    buckets: dict[tuple[tuple[float, ...], ...], list[str]] = {}
    for genotype in genotypes:
        vector = tuple(signatures[(key, genotype)] for key in pair_keys)
        buckets.setdefault(vector, []).append(genotype)
    groups = [sorted(members) for members in buckets.values()]
    groups.sort(key=lambda members: members[0])
    return groups


# --------------------------------------------------------------- 记忆化极小极大

@dataclass(frozen=True)
class TreeNode:
    """极小极大搜索内部使用的不可变树节点。"""

    pair_key: str | None = None
    height: int = 0
    internal_count: int = 0
    members: frozenset[int] = frozenset()
    # (签名块, 子节点) 列表，顺序与规范化分块一致（按块内编号排序）
    branches: tuple[tuple[tuple[float, ...], "TreeNode"], ...] = ()


def canonical_partition(
    blocks_by_signature: dict[tuple[float, ...], frozenset[int]],
) -> tuple[tuple[float, ...], tuple[frozenset[int], ...]]:
    """把“签名 -> 候选块”规范化为可哈希、可去重的分块表示。

    块按块内候选编号升序序列排序，保证同一分块只有一种表示。
    """
    signatures = tuple(sorted(
        blocks_by_signature,
        key=lambda sig: tuple(sorted(blocks_by_signature[sig])),
    ))
    return signatures, tuple(blocks_by_signature[sig] for sig in signatures)


@dataclass(frozen=True)
class _Test:
    """一个酶对实验：pair_key + 对候选全集的规范化分块。"""

    pair_key: str
    signatures: tuple[tuple[float, ...], ...]
    blocks: tuple[frozenset[int], ...]


def optimal_decision_tree(
    candidate_ids: list[int] | frozenset[int],
    tests: list[tuple[str, dict[tuple[float, ...], frozenset[int]]]],
) -> TreeNode:
    """精确求最小化最坏实验次数的自适应决策树（纯函数，便于独立测试）。

    ``tests`` 为 ``(pair_key, {签名: 该签名对应的候选集合})`` 列表；
    分块完全相同的酶对只保留 pair_key 最小者（同构酶对对任何后续选择
    完全等价），只有一个签名块的酶对被剔除。选择次序为最坏深度升序 →
    内部节点数升序 → pair_key 字典序升序；不能拆分当前候选集合的酶对
    被忽略。无任何酶对可拆分全集时返回单叶子。
    """
    universe = frozenset(candidate_ids)

    # 分块去重：同构酶对保留字典序最小的 pair_key
    best_key: dict[tuple[frozenset[int], ...], str] = {}
    sigs_by_partition: dict[tuple[frozenset[int], ...], tuple[tuple[float, ...], ...]] = {}
    for key, blocks_by_signature in tests:
        sigs, blocks = canonical_partition(blocks_by_signature)
        if len(blocks) <= 1:
            continue
        old = best_key.get(blocks)
        if old is None or key < old:
            best_key[blocks] = key
            sigs_by_partition[blocks] = sigs
    unique_tests = sorted(
        (
            _Test(
                pair_key=key,
                signatures=sigs_by_partition[blocks],
                blocks=blocks,
            )
            for blocks, key in best_key.items()
        ),
        key=lambda t: t.pair_key,
    )

    def _leaf(members: frozenset[int]) -> TreeNode:
        return TreeNode(pair_key=None, height=0, internal_count=0, members=members)

    memo: dict[frozenset[int], TreeNode] = {}

    def solve(members: frozenset[int]) -> TreeNode:
        if len(members) <= 1:
            return _leaf(members)

        cached = memo.get(members)
        if cached is not None:
            return cached

        # 在当前集合上能拆分（≥ 2 个非空签名块）的酶对。注意：沿路径已
        # 用过的酶对无需显式剔除——到达本状态时 members 是它某个签名块的
        # 子集，其上签名必然恒定，自然不在此列表中。
        splitting_all = [
            t for t in unique_tests
            if sum(1 for block in t.blocks if block & members) >= 2
        ]

        # 限制划分去重：在 members 上“非空块 -> 该块候选集合”完全相同的酶对
        # 互为孪生——它们在 members 上产生相同分支，且非代表孪生在每个分支
        # 子集合上签名恒定（其区分信息全在孪生块之间），对后代永无用处，
        # 故只需保留 pair_key 字典序最小者作为规范代表；它也决定同深度
        # 破同分结果。代表只取决于 members，记忆化因此可跨实验路径共享。
        restricted: dict[tuple[frozenset[int], ...], _Test] = {}
        for test in splitting_all:
            active = tuple(block for block in test.blocks if block & members)
            incumbent = restricted.get(active)
            if incumbent is None or test.pair_key < incumbent.pair_key:
                restricted[active] = test

        # 块数多的酶对优先尝试以尽早得到较小的上界（仅加速，不改结果）
        splitting = sorted(
            restricted.values(),
            key=lambda t: (-len(tuple(b for b in t.blocks if b & members)), t.pair_key),
        )

        best: TreeNode | None = None
        for test in splitting:
            children: list[tuple[tuple[float, ...], TreeNode]] = []
            for sig, block in zip(test.signatures, test.blocks):
                sub = block & members
                if sub:
                    children.append((sig, solve(sub)))
            height = 1 + max(child.height for _, child in children)
            internal = 1 + sum(child.internal_count for _, child in children)
            candidate = TreeNode(
                pair_key=test.pair_key,
                height=height,
                internal_count=internal,
                members=members,
                branches=tuple(children),
            )
            if best is None or (height, internal, test.pair_key) < (
                best.height, best.internal_count, best.pair_key
            ):
                best = candidate

        # members ≥ 2 但没有任何酶对可拆分：给定测试下这些候选不可分
        # （请求级流程已先按全签名向量合并等价类，实际不会出现这种叶子）
        if best is None:
            best = _leaf(members)
        memo[members] = best
        return best

    return solve(universe)


# --------------------------------------------------------------- 响应组装

def _render_node(node: TreeNode, groups: list[list[str]],
                 pair_lookup: dict[str, list[str]], depth: int) -> DecisionTreeNodeOut:
    if node.pair_key is None:
        return DecisionTreeNodeOut(
            type="leaf",
            depth=depth,
            pair_key=None,
            enzyme_pair=None,
            candidate_count=len(node.members),
            branches=[],
            genotypes=sorted(name for cid in node.members for name in groups[cid]),
        )
    # 分支按签名元组升序（空签名自然最前），与输入顺序无关
    ordered = sorted(node.branches, key=lambda item: item[0])
    return DecisionTreeNodeOut(
        type="internal",
        depth=depth,
        pair_key=node.pair_key,
        enzyme_pair=pair_lookup[node.pair_key],
        candidate_count=len(node.members),
        branches=[
            DecisionTreeBranchOut(
                signature=list(sig),
                child=_render_node(child, groups, pair_lookup, depth + 1),
            )
            for sig, child in ordered
        ],
        genotypes=[],
    )


def build_decision_tree(req: DecisionTreeRequest) -> DecisionTreeResponse:
    """请求级编排：签名 → 不可区分组 → 记忆化极小极大 → 响应模型。"""
    genotypes = sorted({g for enzyme in req.enzymes for g in enzyme.genotype_sites})
    pair_infos, signatures, insufficient = compute_pair_signatures(
        req.enzymes, genotypes,
        req.plasmid_length, req.min_visible_fragment, req.band_resolution_tolerance,
    )
    pair_keys = [key for key, _ in pair_infos]
    pair_lookup = dict(pair_infos)

    groups = build_equivalence_groups(genotypes, pair_keys, signatures)

    # 逐酶对把等价类按签名分块（同组成员签名必然一致，取首个成员代表）
    tests: list[tuple[str, dict[tuple[float, ...], frozenset[int]]]] = []
    for key in pair_keys:
        blocks: dict[tuple[float, ...], set[int]] = {}
        for cid, members in enumerate(groups):
            sig = signatures[(key, members[0])]
            blocks.setdefault(sig, set()).add(cid)
        if len(blocks) >= 2:
            tests.append((key, {sig: frozenset(ids) for sig, ids in blocks.items()}))

    root = optimal_decision_tree(frozenset(range(len(groups))), tests)

    indistinguishable = [members for members in groups if len(members) >= 2]
    pair_signature_out = [
        DecisionTreePairSignatureOut(
            enzyme_pair=pair_lookup[key],
            pair_key=key,
            signatures={
                genotype: list(signatures[(key, genotype)]) for genotype in genotypes
            },
            insufficient_cut_genotypes=insufficient.get(key, []),
        )
        for key in pair_keys
    ]

    if root.pair_key is None and len(groups) > 1:
        # 理论上不可达：≥2 个等价类意味着至少有一个酶对能拆分
        reason = "决策树构造异常"
    elif len(genotypes) == 1:
        reason = (
            f"仅有 1 个基因型（{genotypes[0]}），无需任何双酶切实验即可判定，"
            "决策树为单个叶子，最坏情况下需要 0 次实验。"
        )
    elif len(groups) == 1:
        reason = (
            f"全部 {len(genotypes)} 个基因型在 {len(pair_keys)} 个酶对上的规范化条带签名"
            "完全相同，任何双酶切实验都无法拆分，最坏情况下需要 0 次实验即可判定"
            "（唯一不可区分组见 indistinguishable_groups）。"
        )
    else:
        reason = (
            f"共 {len(genotypes)} 个基因型、{len(pair_keys)} 个酶对；先按全部酶对上的"
            f"规范化签名向量合并出 {len(indistinguishable)} 个不可区分组"
            f"（{len(groups)} 个候选等价类）。记忆化极小极大搜索只考虑能拆分当前候选"
            f"集合的酶对，精确最小化最坏实验次数：最优树根节点选用酶对 {root.pair_key}，"
            f"最坏情况下需要 {root.height} 次实验；同深度方案中该树内部节点最少"
            f"（{root.internal_count} 个内部节点、{root.internal_count + 1} 个叶子），"
            "内部节点数也相同时取根/分支 pair_key 字典序最小者。结果只取决于切点与"
            "酶名、基因型名，与输入提交顺序无关。"
        )

    return DecisionTreeResponse(
        plasmid_length=req.plasmid_length,
        min_visible_fragment=req.min_visible_fragment,
        band_resolution_tolerance=req.band_resolution_tolerance,
        genotype_count=len(genotypes),
        total_enzyme_pairs=len(pair_keys),
        worst_case_depth=root.height,
        internal_node_count=root.internal_count,
        leaf_count=root.internal_count + 1 if root.pair_key is not None else 1,
        indistinguishable_groups=indistinguishable,
        tree=_render_node(root, groups, pair_lookup, 0),
        pair_signatures=pair_signature_out,
        reason=reason,
    )
