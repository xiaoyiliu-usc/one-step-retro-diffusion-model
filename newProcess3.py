# route_per_route_hypergraph_type_guard.py
# -*- coding: utf-8 -*-
"""
逐路径建图（每条完整多级反应路径 = 一条超边）：
- 节点（node）: 路径中出现的所有“分子”节点（仅当 type=="mol" 才纳入）
- 超边（edge）: 整条路径；edge_name/target 仅来自根节点 (type=="mol") 的 name/smiles
- 对任意 type=="reaction" 节点，不读取其 smiles / metadata.smiles / rsmi

输出：
- hypergraph_per_route.pt
    {
      'hyperedge_index': torch.LongTensor(2, R_inc),
      'id_to_mol': List[str],            # node_id -> SMILES（全局去重）
      'routes': [
          {'edge_id': int, 'edge_name': str, 'target': str, 'reaction_count': int}
      ]
    }
- route_incidence.csv   # node_id, smiles, edge_id, edge_name, target_smiles
- route_nodes.csv       # node_id, smiles
- route_edges.csv       # edge_id, edge_name, target, reaction_count
"""

from __future__ import annotations
import json, csv
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import torch


# ---------- 读取 JSON（兼容：单 dict / 列表 / JSON Lines） ----------
def load_routes(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    raw = p.read_text().strip()
    if not raw:
        return []
    # JSON Lines（每行一个 dict）
    if "\n" in raw and raw.lstrip().startswith("{") and raw.count("\n") > 0:
        try:
            routes = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if all(isinstance(x, dict) for x in routes):
                return routes
        except Exception:
            pass
    # 普通 JSON
    data = json.loads(raw)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError("Unrecognized JSON structure.")


# ---------- 仅收集分子节点 + 统计反应步数（用 type 判断） ----------
def gather_molecules_and_reaction_count(route: Dict[str, Any]) -> Tuple[Set[str], int]:
    mols: Set[str] = set()
    rxn_cnt = 0

    def dfs(node: Dict[str, Any]):
        nonlocal rxn_cnt
        if not isinstance(node, dict):
            return
        ntype = node.get("type")
        if ntype == "mol":
            smi = node.get("smiles")
            if isinstance(smi, str) and smi:
                mols.add(smi)
            for ch in (node.get("children") or []):
                dfs(ch)
        elif ntype == "reaction":
            # 反应节点：计数 + 只递归其 children，不读它的 smiles/metadata.smiles/rsmi
            rxn_cnt += 1
            for ch in (node.get("children") or []):
                dfs(ch)
        else:
            # 其他类型（兼容性）
            for ch in (node.get("children") or []):
                dfs(ch)

    dfs(route)
    return mols, rxn_cnt


# ---------- 逐路径=单超边 ----------
def build_hypergraph_per_route(routes: List[Dict[str, Any]]):
    # 1) 全局去重的分子集合；每条路径的分子集合与反应步数
    all_mols: Set[str] = set()
    route_mols_list: List[Set[str]] = []
    route_rxn_counts: List[int] = []
    for r in routes:
        mols, cnt = gather_molecules_and_reaction_count(r)
        route_mols_list.append(mols)
        route_rxn_counts.append(cnt)
        all_mols |= mols

    # 2) 全局稳定编号（按 SMILES 字典序）
    id_to_mol = sorted(all_mols)
    mol_to_id = {s: i for i, s in enumerate(id_to_mol)}

    # 3) incidence 与路由元数据
    top: List[int] = []   # node_id
    bot: List[int] = []   # edge_id
    routes_meta: List[Dict[str, Any]] = []

    for edge_id, (route, mols, rc) in enumerate(zip(routes, route_mols_list, route_rxn_counts)):
        # 仅当根节点 type=="mol" 时，才允许用其 name/smiles 命名
        edge_name = None
        target_smiles = ""
        if isinstance(route, dict) and route.get("type") == "mol":
            # 优先 name；否则 smiles
            raw_name = route.get("name")
            raw_smiles = route.get("smiles")
            if isinstance(raw_name, str) and raw_name.strip():
                edge_name = raw_name.strip()
            elif isinstance(raw_smiles, str) and raw_smiles:
                edge_name = raw_smiles
            if isinstance(raw_smiles, str) and raw_smiles:
                target_smiles = raw_smiles

        if not edge_name:
            edge_name = f"route_{edge_id}"

        # 连接该路径所有分子到此超边
        for smi in mols:
            top.append(mol_to_id[smi])
            bot.append(edge_id)

        routes_meta.append({
            "edge_id": edge_id,
            "edge_name": edge_name,
            "target": target_smiles,
            "reaction_count": int(rc),
        })

    hyperedge_index = torch.tensor([top, bot], dtype=torch.long)
    return hyperedge_index, id_to_mol, routes_meta


# ---------- 保存 CSV ----------
def save_csvs(
    hyperedge_index: torch.Tensor,
    id_to_mol: List[str],
    routes_meta: List[Dict[str, Any]],
    csv_incidence: str | Path = "route_incidence.csv",
    csv_nodes: str | Path = "route_nodes.csv",
    csv_edges: str | Path = "route_edges.csv",
):
    # 节点表
    with open(csv_nodes, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "smiles"])
        for i, s in enumerate(id_to_mol):
            w.writerow([i, s])

    # 超边表（只含根 mol 的 name/smiles）
    with open(csv_edges, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "edge_name", "target", "reaction_count"])
        for e in routes_meta:
            w.writerow([e["edge_id"], e["edge_name"], e["target"], e["reaction_count"]])

    # incidence（node_id -> edge_id），附带 smiles / edge_name / target
    id2name = {e["edge_id"]: e["edge_name"] for e in routes_meta}
    id2target = {e["edge_id"]: e["target"] for e in routes_meta}
    tops = hyperedge_index[0].tolist()
    bots = hyperedge_index[1].tolist()
    with open(csv_incidence, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "smiles", "edge_id", "edge_name", "target_smiles"])
        for n, e in zip(tops, bots):
            w.writerow([n, id_to_mol[n], e, id2name.get(e, f"route_{e}"), id2target.get(e, "")])


# ---------- main ----------
def main(
    input_path: str | Path = "test.json",
    out_pt: str | Path = "hypergraph_per_route.pt",
    csv_incidence: str | Path = "route_incidence.csv",
    csv_nodes: str | Path = "route_nodes.csv",
    csv_edges: str | Path = "route_edges.csv",
):
    routes = load_routes(input_path)
    if not routes:
        raise SystemExit("No routes loaded.")

    hyperedge_index, id_to_mol, routes_meta = build_hypergraph_per_route(routes)

    torch.save(
        {
            "hyperedge_index": hyperedge_index,
            "id_to_mol": id_to_mol,
            "routes": routes_meta,
        },
        out_pt,
    )
    save_csvs(hyperedge_index, id_to_mol, routes_meta, csv_incidence, csv_nodes, csv_edges)

    print("Done.")
    print("nodes:", len(id_to_mol),
          "edges(routes):", len(routes_meta),
          "incidence pairs:", hyperedge_index.shape[1])
    print("Saved:",
          out_pt, ",",
          csv_incidence, ",",
          csv_nodes, ",",
          csv_edges)


if __name__ == "__main__":
    # 默认读取同目录下 test.json
    main("hyperGraph/n1-routes.json",
         "hyperGraph/n1-routes_hypergraph_per_route.pt",
         "hyperGraph/n1-routes_route_incidence.csv",
         "hyperGraph/n1-routes_route_nodes.csv",
         "hyperGraph/n1-routes_route_edges.csv")
