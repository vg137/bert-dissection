import torch.nn.functional as F

def nearest_tokens(word_embedding, n=5, metric='euclidean'):

    if metric == 'euclidean':
        distances = torch.norm(embedding_matrix - word_embedding, p=2, dim=1)
    elif metric == 'cosine':
        distances = F.cosine_similarity(word_embedding, embedding_matrix, dim=1)

    neighbor_distances, neighbor_indices = torch.topk(distances, n, largest=False)
    neighbors = embedding_matrix[neighbor_indices]

    return neighbors, neighbor_distances
