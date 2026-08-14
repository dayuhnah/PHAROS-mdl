import torch
import torch.nn as nn

from src.models.hybrid_drug_encoder import HybridDrugEncoder
from src.models.hybrid_cell_encoder import HybridCellEncoder
from src.models.resistance_film import ResistanceFiLM


class PharosRXDualHybridModel(nn.Module):
    """
    PHAROS V2B

    Drug:
        Morgan fingerprint
            +
        Molecular GINE
            ↓
        adaptive gated fusion
            ↓
          z_drug

    Cell:
        Full gene-expression MLP
            +
        PPI-GNN
            ↓
        adaptive gated fusion
            ↓
          z_cell

    Resistance:
        29 resistance genes
            ↓
        FiLM conditioned on z_drug
            ↓
        z_resistance

    Final:
        z_drug + z_cell + z_resistance
            ↓
        response prediction
    """

    def __init__(
        self,
        drug_fp_dim,
        cell_dim,
        resistance_dim,

        ppi_edge_index,
        ppi_edge_weight,
        num_ppi_genes,

        node_dim=18,
        edge_dim=6,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,
        ppi_hidden_dim=64,

        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        dropout=0.2,
    ):
        super().__init__()

        # ==========================================
        # Hybrid drug encoder
        # ==========================================

        self.drug_encoder = HybridDrugEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            fingerprint_dim=drug_fp_dim,
            gnn_hidden_dim=drug_gnn_hidden_dim,
            out_dim=drug_out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Hybrid cell encoder
        # ==========================================

        self.cell_encoder = HybridCellEncoder(
            full_expression_dim=cell_dim,

            ppi_edge_index=ppi_edge_index,
            ppi_edge_weight=ppi_edge_weight,
            num_ppi_genes=num_ppi_genes,

            expression_hidden_dim=hidden_dim,
            ppi_hidden_dim=ppi_hidden_dim,

            out_dim=cell_out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Resistance FiLM
        # ==========================================

        self.resistance_film = ResistanceFiLM(
            drug_dim=drug_out_dim,
            resistance_dim=resistance_dim,
            hidden_dim=128,
            output_dim=resistance_out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Final fusion
        # ==========================================

        fusion_dim = (
            drug_out_dim
            + cell_out_dim
            + resistance_out_dim
        )

        self.bioactivity_bridge = nn.Sequential(
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

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def forward(
        self,
        drug_graph,
        drug_fp,
        cell_expr,
        ppi_expression,
        resistance_expr,
        return_gates=False,
    ):

        # ==========================================
        # Drug representation
        # ==========================================

        if return_gates:

            (
                z_drug,
                drug_gate,
                z_graph,
                z_fp,
            ) = self.drug_encoder(
                drug_graph,
                drug_fp,
                return_gate=True,
            )

        else:

            z_drug = self.drug_encoder(
                drug_graph,
                drug_fp,
            )

        # ==========================================
        # Cell representation
        # ==========================================

        if return_gates:

            (
                z_cell,
                cell_gate,
                z_expression,
                z_ppi,
            ) = self.cell_encoder(
                cell_expr,
                ppi_expression,
                return_gate=True,
            )

        else:

            z_cell = self.cell_encoder(
                cell_expr,
                ppi_expression,
            )

        # ==========================================
        # Drug-conditioned resistance
        # ==========================================

        z_resistance = self.resistance_film(
            resistance_features=resistance_expr,
            drug_embedding=z_drug,
        )

        # ==========================================
        # Final fusion
        # ==========================================

        fused = torch.cat(
            [
                z_drug,
                z_cell,
                z_resistance,
            ],
            dim=1,
        )

        hidden = self.bioactivity_bridge(
            fused
        )

        prediction = self.response_head(
            hidden
        )

        # ==========================================
        # Optional diagnostic outputs
        # ==========================================

        if return_gates:

            return {
                "prediction":
                    prediction,

                "drug_gate":
                    drug_gate,

                "cell_gate":
                    cell_gate,

                "z_drug":
                    z_drug,

                "z_graph":
                    z_graph,

                "z_fp":
                    z_fp,

                "z_cell":
                    z_cell,

                "z_expression":
                    z_expression,

                "z_ppi":
                    z_ppi,

                "z_resistance":
                    z_resistance,
            }

        return prediction