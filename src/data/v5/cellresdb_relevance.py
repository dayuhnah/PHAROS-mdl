import csv
import json
from pathlib import Path

import torch


class CellResDBRelevanceMapper:
    """
    Build a binary CellResDB token-relevance prior from
    DrugComb drug names.

    The mapping is derived dynamically from:

      1. cellresdb_token_metadata.json
         cohort -> token indices

      2. cellresdb_pathway_tokens.csv
         drug -> cohort evidence

    Nothing depends on hard-coded token indices.
    """

    def __init__(
        self,
        metadata_path,
        pathway_tokens_path,
    ):
        self.metadata_path = Path(
            metadata_path
        )
        self.pathway_tokens_path = Path(
            pathway_tokens_path
        )

        # ====================================================
        # TOKEN METADATA
        # ====================================================

        metadata = json.loads(
            self.metadata_path.read_text()
        )

        tokens = metadata["tokens"]

        self.num_tokens = len(tokens)

        self.cohort_to_tokens = {}

        for token in tokens:
            cohort = str(
                token["cohort_id"]
            )

            token_id = int(
                token["token_id"]
            )

            self.cohort_to_tokens.setdefault(
                cohort,
                [],
            ).append(token_id)

        # ====================================================
        # DRUG -> COHORT EVIDENCE
        # ====================================================

        self.drug_to_cohorts = {}

        with self.pathway_tokens_path.open(
            newline="",
        ) as handle:

            reader = csv.DictReader(
                handle
            )

            for row in reader:

                cohort = str(
                    row["cohort_id"]
                ).strip()

                raw_drugs = (
                    row.get(
                        "matched_pharos_drugs",
                        "",
                    )
                    or ""
                )

                # One CellResDB cohort may contain a regimen
                # represented as:
                #
                # drug1;drug2;drug2;drug3
                #
                # Splitting into a set also removes duplicates.
                drugs = {
                    self.normalize_drug_name(
                        drug
                    )
                    for drug in raw_drugs.split(";")
                    if drug.strip()
                }

                for drug in drugs:

                    self.drug_to_cohorts.setdefault(
                        drug,
                        set(),
                    ).add(cohort)

    @staticmethod
    def normalize_drug_name(name):
        return str(name).strip().lower()

    def matched_cohorts(
        self,
        drug_a,
        drug_b,
    ):
        """
        Return the union of cohorts associated with either
        member of the drug pair.
        """

        drug_a = self.normalize_drug_name(
            drug_a
        )

        drug_b = self.normalize_drug_name(
            drug_b
        )

        cohorts = set()

        cohorts.update(
            self.drug_to_cohorts.get(
                drug_a,
                set(),
            )
        )

        cohorts.update(
            self.drug_to_cohorts.get(
                drug_b,
                set(),
            )
        )

        return cohorts

    def build(
        self,
        drug_a_names,
        drug_b_names,
        device=None,
    ):
        """
        Returns
        -------
        relevance : torch.FloatTensor
            Shape [B, num_real_tokens].

            1 = token belongs to a CellResDB cohort associated
                with Drug A or Drug B.

            0 = no explicit drug-cohort match.
        """

        if len(drug_a_names) != len(
            drug_b_names
        ):
            raise ValueError(
                "drug_a_names and drug_b_names "
                "must have equal length."
            )

        batch_size = len(
            drug_a_names
        )

        relevance = torch.zeros(
            batch_size,
            self.num_tokens,
            dtype=torch.float32,
            device=device,
        )

        for i, (
            drug_a,
            drug_b,
        ) in enumerate(
            zip(
                drug_a_names,
                drug_b_names,
            )
        ):

            cohorts = self.matched_cohorts(
                drug_a,
                drug_b,
            )

            for cohort in cohorts:

                token_ids = (
                    self.cohort_to_tokens.get(
                        cohort,
                        [],
                    )
                )

                if token_ids:
                    relevance[
                        i,
                        token_ids,
                    ] = 1.0

        return relevance
