"""
Medical knowledge graph for graph-based recommendations.

Builds a heterogeneous graph from the project's own data:

    nodes:  disease (41) · symptom (132) · medication · specialist
    edges:  disease—symptom     weight = P(symptom | disease) from training data
            disease—medication  from the curated knowledge base
            disease—specialist  from the curated knowledge base

Two things come out of it:

1. **Graph-walk related diseases** — Personalized PageRank seeded at a disease
   node. Probability mass flows through shared symptoms, medications and
   specialists, so two diseases can be "related" through multi-hop paths even
   when their raw symptom vectors barely overlap. This complements the
   cosine-similarity feature (which only sees direct symptom overlap).

2. **Ego-graph data** for the app's interactive network visualization.

Run standalone to rebuild + smoke-test:  python src/knowledge_graph.py
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
KB_PATH = ROOT / "data" / "processed" / "knowledge_base.json"

# A symptom is linked to a disease when it appears in at least this share of
# that disease's training records.
SYMPTOM_EDGE_THRESHOLD = 0.3


@lru_cache(maxsize=1)
def build_graph() -> nx.Graph:
    """Construct the knowledge graph from the dataset + knowledge base."""
    g = nx.Graph()

    # --- disease—symptom edges from the training data --------------------- #
    df = pd.read_csv(RAW / "disease_symptoms.csv")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(axis=1, how="all")
    df["prognosis"] = df["prognosis"].astype(str).str.strip()
    profiles = df.groupby("prognosis").mean(numeric_only=True)

    for disease, row in profiles.iterrows():
        g.add_node(disease, kind="disease")
        for symptom, p in row.items():
            if p >= SYMPTOM_EDGE_THRESHOLD:
                g.add_node(symptom, kind="symptom")
                g.add_edge(disease, symptom, weight=float(p), rel="has_symptom")

    # --- disease—medication / disease—specialist edges from the KB -------- #
    kb = json.loads(KB_PATH.read_text())
    for disease, info in kb.items():
        g.add_node(disease, kind="disease")
        for med in info.get("medications", []):
            med = med.strip()
            g.add_node(med, kind="medication")
            g.add_edge(disease, med, weight=1.0, rel="treated_by")
        spec = info.get("specialist", "").strip()
        if spec:
            g.add_node(spec, kind="specialist")
            g.add_edge(disease, spec, weight=1.0, rel="consult")

    return g


def graph_related_diseases(disease: str, top_n: int = 5) -> list[dict]:
    """Related diseases via Personalized PageRank from the given disease node.

    Returns [{disease, score}, ...] — score is the PageRank mass that flowed
    to the other disease through shared symptoms/medications/specialists.
    """
    g = build_graph()
    if disease not in g:
        return []
    ppr = nx.pagerank(g, personalization={disease: 1.0}, weight="weight")
    ranked = [
        (n, s)
        for n, s in sorted(ppr.items(), key=lambda kv: kv[1], reverse=True)
        if g.nodes[n]["kind"] == "disease" and n != disease
    ][:top_n]
    return [{"disease": n, "score": round(float(s), 5)} for n, s in ranked]


def ego_graph_data(disease: str, max_neighbors: int = 14) -> dict | None:
    """Nodes + edges of the disease's immediate neighborhood, for plotting.

    Neighbors are capped (strongest edges first) so the visualization stays
    readable. Returns {nodes: [{id, kind}], edges: [{source, target, weight, rel}]}.
    """
    g = build_graph()
    if disease not in g:
        return None

    neighbors = sorted(
        g[disease].items(), key=lambda kv: kv[1].get("weight", 0), reverse=True
    )[:max_neighbors]

    nodes = [{"id": disease, "kind": "disease"}]
    edges = []
    for n, attrs in neighbors:
        nodes.append({"id": n, "kind": g.nodes[n]["kind"]})
        edges.append(
            {
                "source": disease,
                "target": n,
                "weight": round(float(attrs.get("weight", 1.0)), 3),
                "rel": attrs.get("rel", ""),
            }
        )
    return {"nodes": nodes, "edges": edges}


def graph_stats() -> dict:
    g = build_graph()
    kinds = pd.Series([d["kind"] for _, d in g.nodes(data=True)]).value_counts()
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        **{f"n_{k}": int(v) for k, v in kinds.items()},
    }


if __name__ == "__main__":
    print("Graph:", graph_stats())
    for d in ["Migraine", "Hepatitis B", "Diabetes "]:
        rel = graph_related_diseases(d.strip(), top_n=3)
        print(f"\nGraph-related to {d.strip()}:")
        for r in rel:
            print(f"   {r['disease']:<30} {r['score']}")
    ego = ego_graph_data("Migraine")
    print(f"\nMigraine ego-graph: {len(ego['nodes'])} nodes, {len(ego['edges'])} edges")
