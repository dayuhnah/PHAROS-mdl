import torch
import torch.nn as nn

from src.models.molecular_gnn_encoder import (
    MolecularGNNEncoder,
)

from src.models.ppi_cell_gnn import (
    PPICellGNNEncoder,
)

from src.models.resistance_film import (
    ResistanceFiLM,
)


class PharosDualGNNModel(nn.Module):
    """
    PHAROS with:

    1. Molecular drug GNN
    2. PPI cancer-cell GNN
    3. Drug-conditioned resistance FiLM
    """

    def __init__(
        self,
        ppi_edge_index,
        ppi_edge_weight,
        num_ppi_genes,
        resistance_dim,

        node_dim=18,
        edge_dim=6,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,
        cell_gnn_hidden_dim=64,

        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        dropout=0.2,
    ):
        super().__init__()

        # =================================
        # Drug molecular GNN
        # =================================

        self.drug_encoder = (
            MolecularGNNEncoder(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=(
                    drug_gnn_hidden_dim
                ),
                out_dim=drug_out_dim,
                dropout=dropout,
            )
        )

        # =================================
        # Cell PPI GNN
        # =================================

        self.cell_encoder = (
            PPICellGNNEncoder(
                edge_index=(
                    ppi_edge_index
                ),
                edge_weight=(
                    ppi_edge_weight
                ),
                num_genes=(
                    num_ppi_genes
                ),
                gene_embedding_dim=16,
                hidden_dim=(
                    cell_gnn_hidden_dim
                ),
                out_dim=cell_out_dim,
                dropout=dropout,
            )
        )

        # =================================
        # Resistance conditioning
        # =================================

        self.resistance_film = (
            ResistanceFiLM(
                drug_dim=drug_out_dim,
                resistance_dim=(
                    resistance_dim
                ),
                hidden_dim=128,
                output_dim=(
                    resistance_out_dim
                ),
                dropout=dropout,
            )
        )

        # =================================
        # Fusion
        # =================================

        fusion_dim = (
            drug_out_dim
            + cell_out_dim
            + resistance_out_dim
        )

        self.bioactivity_bridge = (
            nn.Sequential(
                nn.Linear(
                    fusion_dim,
                    hidden_dim,
                ),
                nn.ReLU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    hidden_dim,
                    hidden_dim // 2,
                ),
                nn.ReLU(),
            )
        )

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def forward(
        self,
        drug_graph,
        ppi_expression,
        resistance_expr,
    ):

        # ---------------------------------
        # Drug
        # ---------------------------------

        z_drug = self.drug_encoder(
            x=drug_graph.x,
            edge_index=(
                drug_graph.edge_index
            ),
            edge_attr=(
                drug_graph.edge_attr
            ),
            batch=drug_graph.batch,
        )

        # ---------------------------------
        # Cancer cell
        # ---------------------------------

        z_cell = self.cell_encoder(
            ppi_expression
        )

        # ---------------------------------
        # Resistance
        # ---------------------------------

        z_resistance = (
            self.resistance_film(
                resistance_features=(
                    resistance_expr
                ),
                drug_embedding=z_drug,
            )
        )

        # ---------------------------------
        # Fusion
        # ---------------------------------

        z = torch.cat(
            [
                z_drug,
                z_cell,
                z_resistance,
            ],
            dim=1,
        )

        z = self.bioactivity_bridge(
            z
        )

        return self.response_head(
            z
        )