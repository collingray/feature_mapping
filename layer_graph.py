import plotly.graph_objects as go
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F


def grid_map(i: int, size: int):
    return i % size, i // size


def zaxis_scurve(dx, dy, dz, steps=10):
    unit_steps = np.linspace(0, 1, steps)
    unit_curve = -2 * unit_steps ** 3 + 3 * unit_steps ** 2

    x1, x2 = dx
    y1, y2 = dy
    z1, z2 = dz

    x = x1 * (1 - unit_curve) + x2 * unit_curve
    y = y1 * (1 - unit_curve) + y2 * unit_curve
    z = np.linspace(z1, z2, steps)

    return x, y, z


def calc_gradient(color1, color2, steps=10):
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    r = torch.linspace(r1, r2, steps)
    g = torch.linspace(g1, g2, steps)
    b = torch.linspace(b1, b2, steps)

    return torch.stack([r, g, b], dim=1)


def plot_layer_graph(data: torch.Tensor, threshold: float = 0.5):
    """
    :param data: [num_layers-1, num_features, num_features]
    :param threshold:
    """

    features = data.shape[1]

    x_dim = np.ceil(np.sqrt(features))
    y_dim = np.ceil(features / x_dim)
    z_dim = data.shape[0]

    fig = go.Figure()

    xy_offset = 0.5
    for i in range(z_dim + 1):
        fig.add_trace(go.Surface(
            z=[[z_dim - i, z_dim - i], [z_dim - i, z_dim - i]],
            x=[[-xy_offset, x_dim - xy_offset], [-xy_offset, x_dim - xy_offset]],
            y=[[-xy_offset, -xy_offset], [y_dim - xy_offset, y_dim - xy_offset]],
            colorscale='Viridis',
            showscale=False,
            opacity=0.3,
            name=f'Layer {i}',
        ))

    connections = torch.where(data >= threshold, data, torch.zeros_like(data))
    connections = connections / connections.max()
    norm_connections = connections / connections.norm(p=1, dim=1, keepdim=True)

    colors = (torch.tensor(plt.get_cmap("rainbow")(np.linspace(0, 1, features)))[:, :3]
              .to(data.dtype).repeat(z_dim, 1, 1))

    for i in range(z_dim - 1):
        next_colors = norm_connections[i].T @ colors[i]
        next_colors = torch.where(next_colors.sum(dim=1, keepdim=True) > 0, next_colors, colors[0])
        colors[i + 1] = next_colors

    steps = 10
    for i, j, k in zip(*torch.where(data >= threshold)):
        x1, y1 = grid_map(j, x_dim)
        x2, y2 = grid_map(k, x_dim)
        z1, z2 = z_dim - i, z_dim - i - 1
        x, y, z = zaxis_scurve((x1, x2), (y1, y2), (z1, z2), steps)
        gradient = calc_gradient(colors[i][j], colors[i][k], steps)

        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color=gradient, width=2),
            name=f'{j} -> {k}',
            opacity=connections[i, j, k].item(),
        ))

    scale = 0.5
    layer_height_to_width = 0.5

    fig.update_layout(
        scene=dict(
            xaxis_title='Feature',
            yaxis_title='Feature',
            zaxis_title='Layer',
            aspectmode='manual',
            aspectratio=dict(x=scale, y=scale * (y_dim / x_dim), z=scale * layer_height_to_width * z_dim),
        ),
        title='Layer Graph',
    )

    return fig


def plot_feature_trace(feature: int, data: torch.Tensor, top_feats=4, top_conns=100, threshold: float = 0.5):
    """
    :param feature:
    :param data: [num_layers-1, num_features, num_features]
    :param threshold:
    """

    n_features = data.shape[1]

    x_dim = np.ceil(np.sqrt(n_features))
    y_dim = np.ceil(n_features / x_dim)
    z_dim = data.shape[0]

    fig = go.Figure()
    fig.layout.height = 1200

    xy_offset = 0.555
    for l in range(z_dim + 1):
        fig.add_trace(go.Surface(
            z=[[z_dim - l, z_dim - l], [z_dim - l, z_dim - l]],
            x=[[-xy_offset, x_dim - xy_offset], [-xy_offset, x_dim - xy_offset]],
            y=[[-xy_offset, -xy_offset], [y_dim - xy_offset, y_dim - xy_offset]],
            colorscale='Viridis',
            showscale=False,
            opacity=0.3,
            name=f'Layer {l}',
        ))

    # connections = torch.where(data >= threshold, data, torch.zeros_like(data))
    # connections = connections / connections.max()
    # norm_connections = connections / connections.norm(p=1, dim=1, keepdim=True)
    #
    # colors = torch.tensor(plt.get_cmap("rainbow")(np.linspace(0, 1, features)))[:, :3].to(data.dtype).repeat(z_dim, 1,
    #                                                                                                          1)
    #
    # for l in range(z_dim - 1):
    #     next_colors = norm_connections[l].T @ colors[l]
    #     next_colors = torch.where(next_colors.sum(dim=1, keepdim=True) > 0, next_colors, colors[0])
    #     colors[l + 1] = next_colors

    curr_feats = F.one_hot(torch.tensor([feature]), n_features).squeeze().to(device="cuda", dtype=torch.bfloat16)
    connections = []

    for i in range(z_dim):
        next_feats = data[-i - 1].T @ curr_feats
        values, next_indices = torch.topk(next_feats, k=top_feats)

        # indices of all activated features on the current layer
        curr_indices = torch.where(curr_feats > 0)[0]

        # all connections from current features to next features, weighted by the strength of the current features
        possible_connections = data[-i, curr_indices, next_indices] * curr_feats[curr_indices].unsqueeze(1)

        # indices of the top connections
        _, top_connection_idxs = torch.topk(possible_connections.flatten(),
                                            k=min(top_conns, possible_connections.flatten().shape[0]))
        c, n = torch.unravel_index(top_connection_idxs, possible_connections.shape)
        curr_conn_indices = curr_indices[c]
        next_conn_indices = next_indices[n]
        chosen_connections = torch.stack([curr_conn_indices, next_conn_indices], dim=1)

        connections = chosen_connections.unsqueeze(dim=0).tolist() + connections
        curr_feats[:] = 0
        curr_feats[next_indices] = values / values.sum()

    steps = 10
    for layer in range(z_dim):
        for f1, f2 in connections[layer]:
            x1, y1 = grid_map(f1, x_dim)
            x2, y2 = grid_map(f2, x_dim)
            z1, z2 = z_dim - layer - 1, z_dim - layer
            x, y, z = zaxis_scurve((x1, x2), (y1, y2), (z1, z2), steps)
            gradient = "blue"

            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line=dict(color=gradient, width=2),
                name=f'{f1} -> {f2}',
                opacity=1  # connections[layer, f1, f2].item(),
            ))

    scale = 0.35
    layer_height_to_width = 0.5

    fig.update_layout(
        scene=dict(
            xaxis_title='Feature X',
            yaxis_title='Feature Y',
            zaxis_title='Layer',
            xaxis=dict(tickvals=[]),
            yaxis=dict(tickvals=[]),
            zaxis=dict(tickvals=[]),
            aspectmode='manual',
            aspectratio=dict(x=scale, y=scale * (y_dim / x_dim), z=scale * layer_height_to_width * z_dim),
        ),
        title='Layer Graph',
    )

    return fig
