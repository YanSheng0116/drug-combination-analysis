import networkx as nx
from sklearn.preprocessing import normalize
import numpy as np

def random_walk_with_restart(adj_matrix, seeds, alpha=0.5, max_iter=100):
    """
    adj_matrix: STRING PPI邻接矩阵
    seeds: 初始药物靶点向量
    alpha: 重启概率
    """
    prob = seeds.copy()
    for _ in range(max_iter):
        prob_new = alpha * np.dot(adj_matrix, prob) + (1 - alpha) * seeds
        if np.allclose(prob, prob_new):
            break
        prob = prob_new
    return normalize(prob.reshape(1, -1))[0]