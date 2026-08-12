from __future__ import annotations

import argparse
from pathlib import Path

GENERATION_MARKER = "# STACKPILOT_QUERY_CREDIT_GENERATION_V1"
TRAINER_MARKER = "# STACKPILOT_QUERY_CREDIT_TRAINER_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def patch_generation(root: Path) -> None:
    target = root / "search_r1" / "llm_agent" / "generation.py"
    text = target.read_text(encoding="utf-8")
    if GENERATION_MARKER in text:
        required = (
            "build_search_span_ids(",
            "self._stackpilot_search_document_batches",
            "'stackpilot_search_document_batches'",
            "'stackpilot_search_span_ids'",
        )
        missing = [value for value in required if value not in text]
        if missing:
            raise RuntimeError(f"Incomplete query-credit generation patch: {missing}")
        print(f"Query-credit generation patch already present: {target}")
        return
    if "# STACKPILOT_BEHAVIOR_QUOTIENT_GENERATION_V1" not in text:
        raise RuntimeError("Apply behavior-quotient generation patch before query-credit")
    text = replace_once(
        text,
        "from stackpilot.action_protocol import parse_action\n",
        "from stackpilot.action_protocol import parse_action\n"
        "from stackpilot.query_credit_runtime import build_search_span_ids\n"
        f"{GENERATION_MARKER}\n",
        "query-credit runtime import",
    )
    text = replace_once(
        text,
        "        self._stackpilot_search_title_batches = [\n"
        "            [] for _ in range(protocol_batch_size)\n"
        "        ]\n",
        "        self._stackpilot_search_title_batches = [\n"
        "            [] for _ in range(protocol_batch_size)\n"
        "        ]\n"
        "        self._stackpilot_search_document_batches = [\n"
        "            [] for _ in range(protocol_batch_size)\n"
        "        ]\n",
        "query-credit protocol document state",
    )
    text = replace_once(
        text,
        "        self._stackpilot_last_search_titles = []\n",
        "        self._stackpilot_last_search_titles = []\n"
        "        self._stackpilot_last_search_documents = []\n",
        "raw document metadata reset",
    )
    text = replace_once(
        text,
        "            titles = []\n",
        "            titles = []\n            documents = []\n",
        "raw document metadata list",
    )
    text = replace_once(
        text,
        "                titles.append(title)\n",
        "                titles.append(title)\n"
        "                text = contents.split('\\n', 1)[1] if '\\n' in contents else ''\n"
        "                raw_score = item.get('score', item.get('retrieval_score', 0.0))\n"
        "                try:\n"
        "                    numeric_score = float(raw_score)\n"
        "                except (TypeError, ValueError):\n"
        "                    numeric_score = 0.0\n"
        "                documents.append({\n"
        "                    'document_rank': document_index + 1,\n"
        "                    'document_title': title,\n"
        "                    'document_text': text,\n"
        "                    'retriever_score': numeric_score,\n"
        "                })\n",
        "raw document metadata extraction",
    )
    text = replace_once(
        text,
        "            self._stackpilot_last_search_titles.append(titles)\n",
        "            self._stackpilot_last_search_titles.append(titles)\n"
        "            self._stackpilot_last_search_documents.append(documents)\n",
        "raw document metadata return",
    )
    setup_anchor = (
        "            search_title_batches = [\n"
        "                list(titles) for titles in search_title_batches\n"
        "            ]\n"
    )
    setup_replacement = setup_anchor + (
        "            search_document_batches = getattr(\n"
        "                self, '_stackpilot_last_search_documents', None\n"
        "            )\n"
        "            if (\n"
        "                not isinstance(search_document_batches, list)\n"
        "                or len(search_document_batches) != len(search_results)\n"
        "            ):\n"
        "                raise RuntimeError(\n"
        "                    'retriever document metadata does not match search results'\n"
        "                )\n"
        "            search_document_batches = [\n"
        "                [dict(document) for document in documents]\n"
        "                for documents in search_document_batches\n"
        "            ]\n"
    )
    text = replace_once(text, setup_anchor, setup_replacement, "document batch setup")
    text = replace_once(
        text,
        "        else:\n            search_title_batches = [[] for _ in search_results]\n",
        "        else:\n"
        "            search_title_batches = [[] for _ in search_results]\n"
        "            search_document_batches = [[] for _ in search_results]\n",
        "forced-final document placeholders",
    )
    text = replace_once(
        text,
        "                    current_search_titles = list(search_title_batches.pop(0))\n",
        "                    current_search_titles = list(search_title_batches.pop(0))\n"
        "                    current_search_documents = [\n"
        "                        dict(document)\n"
        "                        for document in search_document_batches.pop(0)\n"
        "                    ]\n",
        "consume document metadata",
    )
    text = replace_once(
        text,
        "                    self._stackpilot_search_title_batches[i].append(\n"
        "                        current_search_titles\n"
        "                    )\n",
        "                    self._stackpilot_search_title_batches[i].append(\n"
        "                        current_search_titles\n"
        "                    )\n"
        "                    self._stackpilot_search_document_batches[i].append(\n"
        "                        current_search_documents\n"
        "                    )\n",
        "record trajectory documents",
    )
    text = replace_once(
        text,
        "                    search_observed_title_batches.pop(0)\n"
        "                    next_obs.append('')\n",
        "                    search_observed_title_batches.pop(0)\n"
        "                    search_document_batches.pop(0)\n"
        "                    next_obs.append('')\n",
        "consume forced-final document metadata",
    )
    text = replace_once(
        text,
        "        assert len(search_observed_title_batches) == 0\n",
        "        assert len(search_observed_title_batches) == 0\n"
        "        assert len(search_document_batches) == 0\n",
        "document metadata exhaustion",
    )
    text = replace_once(
        text,
        "            'stackpilot_search_title_batches': self._stackpilot_search_title_batches,\n",
        "            'stackpilot_search_title_batches': self._stackpilot_search_title_batches,\n"
        "            'stackpilot_search_document_batches': self._stackpilot_search_document_batches,\n",
        "document protocol output",
    )
    span_block = (
        "        final_output['stackpilot_search_span_ids'] = build_search_span_ids(\n"
        "            final_output['responses'],\n"
        "            pad_token_id=int(self.tokenizer.pad_token_id),\n"
        "            open_ids=self.tokenizer(\n"
        "                '<search>', add_special_tokens=False\n"
        "            )['input_ids'],\n"
        "            close_ids=self.tokenizer(\n"
        "                '</search>', add_special_tokens=False\n"
        "            )['input_ids'],\n"
        "        )\n\n"
    )
    text = replace_once(
        text,
        "        protocol_values = {\n",
        span_block + "        protocol_values = {\n",
        "search-span tensor",
    )
    target.write_text(text, encoding="utf-8")
    print(f"Applied query-credit generation patch: {target}")


def patch_trainer(root: Path) -> None:
    target = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = target.read_text(encoding="utf-8")
    if TRAINER_MARKER in text:
        required = (
            "apply_query_credit_bonus(",
            "STACKPILOT_QC_MODE",
            "stackpilot_search_span_ids",
            "stackpilot_search_document_batches",
        )
        missing = [value for value in required if value not in text]
        if missing:
            raise RuntimeError(f"Incomplete query-credit trainer patch: {missing}")
        print(f"Query-credit trainer patch already present: {target}")
        return
    if "# STACKPILOT_BEHAVIOR_QUOTIENT_TRAINER_V1" not in text:
        raise RuntimeError("Apply behavior-quotient trainer patch before query-credit")
    import_anchor = (
        "from stackpilot.behavior_quotient_runtime import (\n"
        "    append_behavior_telemetry,\n"
        "    compute_behavior_advantages,\n"
        "    select_behavior_rows,\n"
        ")\n"
    )
    text = replace_once(
        text,
        import_anchor,
        import_anchor
        + f"{TRAINER_MARKER}\n"
        + "from stackpilot.query_credit_runtime import apply_query_credit_bonus\n",
        "query-credit trainer import",
    )
    call_anchor = "        data.batch['stackpilot_bq_selected_mask'] = bq_selected_mask\n"
    call_block = (
        "        qc_mode = os.environ.get('STACKPILOT_QC_MODE', 'outcome')\n"
        "        qc_rows = []\n"
        "        qc_metrics = {}\n"
        "        if qc_mode != 'outcome':\n"
        "            search_span_ids = data.batch.get('stackpilot_search_span_ids')\n"
        "            document_batches = data.non_tensor_batch.get(\n"
        "                'stackpilot_search_document_batches'\n"
        "            )\n"
        "            if search_span_ids is None or document_batches is None:\n"
        "                raise RuntimeError(\n"
        "                    'query-credit shaping requires search-span and document metadata'\n"
        "                )\n"
        "            advantages, qc_metrics, qc_rows = apply_query_credit_bonus(\n"
        "                advantages=advantages,\n"
        "                search_span_ids=search_span_ids,\n"
        "                index=list(index),\n"
        "                query_batches=list(query_batches),\n"
        "                title_batches=list(title_batches),\n"
        "                document_batches=list(document_batches),\n"
        "                mode=qc_mode,\n"
        "                model_path=os.environ['STACKPILOT_QC_MODEL'],\n"
        "                backend=os.environ.get('STACKPILOT_BQ_BACKEND', 'unknown'),\n"
        "                aggregation=os.environ.get(\n"
        "                    'STACKPILOT_QC_AGGREGATION', 'positive-sum'\n"
        "                ),\n"
        "                alpha=float(os.environ.get('STACKPILOT_QC_ALPHA', '0.25')),\n"
        "                seed=int(os.environ.get('STACKPILOT_BQ_RUN_SEED', '0')),\n"
        "            )\n"
        "        data.meta_info['stackpilot_qc_metrics'] = qc_metrics\n"
        "        data.meta_info['stackpilot_qc_rows'] = qc_rows\n"
    )
    text = replace_once(text, call_anchor, call_block + call_anchor, "query-credit advantage call")
    telemetry_anchor = "                        bq_rows = batch.meta_info.pop('stackpilot_bq_rows', None)\n"
    telemetry_block = telemetry_anchor + (
        "                        qc_metrics = batch.meta_info.pop(\n"
        "                            'stackpilot_qc_metrics', None\n"
        "                        )\n"
        "                        if isinstance(qc_metrics, dict):\n"
        "                            metrics.update(qc_metrics)\n"
        "                        qc_rows = batch.meta_info.pop(\n"
        "                            'stackpilot_qc_rows', None\n"
        "                        )\n"
        "                        qc_path = os.environ.get(\n"
        "                            'STACKPILOT_QC_TELEMETRY_PATH', ''\n"
        "                        ).strip()\n"
        "                        if qc_path and isinstance(qc_rows, list):\n"
        "                            append_behavior_telemetry(\n"
        "                                qc_path,\n"
        "                                global_step=self.global_steps,\n"
        "                                rows=qc_rows,\n"
        "                                metadata={\n"
        "                                    'experiment_id': os.environ.get(\n"
        "                                        'STACKPILOT_EXPERIMENT_ID', 'unknown'\n"
        "                                    ),\n"
        "                                    'variant': os.environ.get(\n"
        "                                        'STACKPILOT_EXPERIMENT_VARIANT', 'unknown'\n"
        "                                    ),\n"
        "                                    'credit_mode': os.environ.get(\n"
        "                                        'STACKPILOT_QC_MODE', 'outcome'\n"
        "                                    ),\n"
        "                                },\n"
        "                            )\n"
    )
    text = replace_once(text, telemetry_anchor, telemetry_block, "query-credit telemetry")
    target.write_text(text, encoding="utf-8")
    print(f"Applied query-credit trainer patch: {target}")


def patch(search_r1_root: Path) -> None:
    patch_generation(search_r1_root)
    patch_trainer(search_r1_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-r1-root", required=True)
    args = parser.parse_args()
    patch(Path(args.search_r1_root).resolve())
