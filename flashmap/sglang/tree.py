# coding=utf-8
"""The verify tree of a multi-candidate FlashMap draft (pure torch, no sglang
imports)."""

import torch
import torch.nn.functional as F


def build_tree_abi(candidates: torch.Tensor, width: int):
    """The verify tree from N candidate blocks per request, in EAGLE's encoding.

    ``candidates (bs, N, K)`` are the drafts after each request's anchor (the
    tree's root). Their trie has a node per distinct prefix; a node's vote is
    how many candidates pass through it, a Monte Carlo estimate of the prefix's
    probability under the map that never grows with depth. The ``width - 1``
    nodes with the most votes (ties: shallower, then lower candidate) are kept,
    so every kept node's parent is kept too, and listed parents first (by depth,
    then candidate). A trie with fewer nodes is padded with copies of the first
    node under the root, listed after it: verification takes the first matching
    child, so a copy is never accepted.

    Returns ``tokens (bs, width - 1)``; ``ids (bs, width - 1)``, node n's id
    ``parent(n) * width + n`` (root = 0), which build_tree_kernel_efficient
    decodes with topk = width and parent_list[:, n] = ids[:, n - 1]; and
    ``stats (bs, 2)``: the trie's node count before the cut and the number of
    distinct candidates.
    All on device, fixed shapes, no host synchronization.
    """
    bs, n, k = candidates.shape
    m = width - 1
    device = candidates.device
    # eq[b, i, j, d]: candidates i and j agree on positions 0..d.
    eq = (
        (candidates.unsqueeze(2) == candidates.unsqueeze(1))
        .to(torch.int32)
        .cumprod(dim=-1)
    )
    votes = eq.sum(dim=2)  # (bs, N, K)
    first = eq.argmax(dim=2)  # the lowest candidate with the same prefix
    cand = torch.arange(n, device=device).view(1, n, 1)
    depth = torch.arange(k, device=device).view(1, 1, k)
    real = first == cand  # one representative slot per trie node
    key = torch.where(
        real, votes * (n * k) + (k - 1 - depth) * n + (n - 1 - cand), -1
    ).view(bs, n * k)
    if n * k < m:
        key = F.pad(key, (0, m - n * k), value=-1)
    # Keys of real nodes are distinct, so a sort ranks exactly as topk would;
    # topk's radix select costs ~15 us per step here against ~4 us for a sort.
    top, slot = key.sort(dim=1, descending=True)
    top, slot = top[:, :m], slot[:, :m]
    pad = top < 0
    slot = slot.clamp(max=n * k - 1)
    i, d = slot // k, slot % k
    order = torch.where(
        pad, n * k + torch.arange(m, device=device).view(1, m), d * n + i
    )
    order, perm = order.sort(dim=1)
    slot, pad, i, d = (
        slot.gather(1, perm),
        pad.gather(1, perm),
        i.gather(1, perm),
        d.gather(1, perm),
    )
    index = torch.arange(1, m + 1, device=device).view(1, m).expand(bs, m)
    # Node index of each kept slot; padding goes to a spare last column.
    node_of = torch.zeros(bs, n * k + 1, dtype=torch.int64, device=device)
    node_of.scatter_(1, torch.where(pad, n * k, slot), index)
    # A node's parent is the representative of its prefix one shorter.
    up = first.view(bs, n * k).gather(1, (slot - 1).clamp(min=0)) * k + (d - 1).clamp(
        min=0
    )
    parent = torch.where((d == 0) | pad, 0, node_of.gather(1, up))
    tokens = candidates.reshape(bs, n * k).gather(1, slot)
    tokens = torch.where(pad, tokens[:, :1], tokens)
    distinct = real[:, :, k - 1].sum(dim=1)  # candidates that are their own last node
    stats = torch.stack([real.view(bs, n * k).sum(dim=1), distinct], dim=1)
    return tokens, parent * width + index, stats
