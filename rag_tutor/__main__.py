import sys
import argparse
import json as _json
from pathlib import Path

def run_ui():
    """Launches the Streamlit UI programmatically."""
    try:
        from streamlit.web import cli as stcli
    except ImportError as e:
        print(f"Error: {e}")
        print("Please ensure you are running Python from your virtual environment.")
        print("Try running: ./venv/bin/python -m rag_tutor run")
        sys.exit(1)
        
    current_dir = Path(__file__).parent
    app_path = current_dir / "ui" / "streamlit_app.py"
    
    # Configure arguments to pass to streamlit
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())

def run_ingest(subject_name: str):
    """Executes the document ingestion CLI pipeline."""
    from rag_tutor.config import STUDY_DIR
    from rag_tutor.ingestion import ingest_pdfs
    
    pdf_directory = STUDY_DIR / subject_name
    pdf_directory.mkdir(parents=True, exist_ok=True)
    print(f"Running CLI ingestion for subject '{subject_name}' in: {pdf_directory}")
    try:
        processed = ingest_pdfs(pdf_directory, subject_name)
        print(f"Ingestion completed. Processed {processed} files.")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)

def run_eval(dataset_path: str | None, subject: str | None, k: int, as_json: bool, leakage: bool):
    """Run the RAG retrieval evaluation harness.

    Measures Hit Rate, MRR, Precision@k, Recall@k over a ground-truth dataset
    using the live retrieval pipeline. When no ``--subject`` override is given,
    each query is routed to ITS OWN subject (per-item), so multi-subject
    datasets are evaluated correctly. With ``--leakage``, also reports
    cross-subject contamination when subject filtering is disabled. Eval is
    best-effort: a missing subject/API key prints an actionable message and
    exits 0 rather than failing.
    """
    # Lazy imports keep `import rag_tutor` light.
    from rag_tutor.eval.dataset import load_dataset
    from rag_tutor.eval.runner import run_eval, run_leakage_eval
    from rag_tutor.retrieval import retrieve_context, has_subject_documents

    dataset = load_dataset(dataset_path)
    if not dataset:
        if dataset_path:
            print(f"No valid items found in dataset '{dataset_path}'.")
        else:
            print("The built-in default dataset is empty.")
        print("See eval/dataset.example.json for the expected format.")
        sys.exit(0)

    # When a subject override is given, every query is forced into it.
    # Otherwise each query uses its own per-item subject (multi-subject aware).
    override = subject
    subjects_in_use = (
        {override} if override else {it.subject for it in dataset if it.subject}
    )

    # Pre-flight: every subject we'll touch must have ingested documents.
    missing = [s for s in sorted(subjects_in_use) if not has_subject_documents(s)]
    if missing:
        msg = (
            f"No documents ingested for subject(s): {', '.join(repr(s) for s in missing)}. "
            "Run: python -m rag_tutor ingest --subject \"<subject>\""
        )
        if as_json:
            print(_json.dumps({"error": msg, "missing_subjects": missing}))
        else:
            print(msg)
        sys.exit(0)

    def retrieve(query, subj, depth):
        return retrieve_context(query=query, subject=subj, n_results=depth)

    # subject=None -> runner respects each item's own subject.
    report = run_eval(dataset, retrieve, k=k, subject=override)

    # Optional cross-subject leakage probe.
    leakage_report = None
    if leakage and subjects_in_use:
        leakage_report = _run_leakage_probe(dataset, k=k, override=override)

    _emit_eval_output(report, leakage_report, override, subjects_in_use, as_json)


def _run_leakage_probe(dataset, k: int, override: str | None):
    """Query WITHOUT subject filtering and measure cross-subject contamination.

    Returns a LeakageReport. Each retrieved chunk's own subject is read from
    the raw Chroma metadata so we can tell whether subject A's query pulled in
    subject B's chunks.
    """
    from rag_tutor.eval.runner import run_leakage_eval
    from rag_tutor.llm.embeddings import get_embedding
    from rag_tutor.retrieval.query_expansion import expand_query
    from rag_tutor.retrieval.vector_store import get_collection

    collection = get_collection()

    def cross_subject(query, depth):
        # Embed + query the store with NO subject where-clause.
        emb = get_embedding(expand_query(query))
        res = collection.query(query_embeddings=[emb], n_results=depth)
        rows = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for doc, meta in zip(docs, metas):
            rows.append((
                meta.get("source", "?"),
                meta.get("page", 0),
                meta.get("subject"),
            ))
        return rows

    return run_leakage_eval(dataset, cross_subject, k=k)


def _emit_eval_output(report, leakage_report, override, subjects_in_use, as_json):
    """Format and print the retrieval + (optional) leakage results."""
    subject_label = (
        override if override
        else (next(iter(subjects_in_use)) if len(subjects_in_use) == 1 else "multiple (per-item)")
    )

    if as_json:
        out = report.to_dict()
        out["subject"] = subject_label
        out["items"] = [
            {
                "query": it.query,
                "subject": it.subject,
                "hit": it.hit,
                "reciprocal_rank": it.rr,
                "precision": it.precision,
                "recall": it.recall,
                "n_retrieved": it.n_retrieved,
                "error": it.error,
            }
            for it in report.items
        ]
        if leakage_report is not None:
            out["leakage"] = leakage_report.to_dict()
        print(_json.dumps(out, indent=2))
        return

    _print_report_table(report, subject_label)
    if leakage_report is not None:
        _print_leakage_table(leakage_report)



def _print_report_table(report, subject: str | None):
    """Pretty-print the eval metrics as a table."""
    subj_label = subject or "(all)"
    print()
    print(f"  Retrieval Evaluation — subject: {subj_label} (k={report.k})")
    print()
    rows = [
        ("Hit Rate", f"{report.hit_rate:.2%}"),
        ("MRR", f"{report.mrr:.4f}"),
        (f"Precision@{report.k}", f"{report.precision_at_k:.2%}"),
        (f"Recall@{report.k}", f"{report.recall_at_k:.2%}"),
        ("Items evaluated", str(report.n_items)),
    ]
    if report.n_errors:
        rows.append(("Retrieval errors", str(report.n_errors)))
    label_w = max(len(r[0]) for r in rows)
    top = f"  ┌{'─' * (label_w + 2)}┬{'─' * 12}┐"
    mid = f"  ├{'─' * (label_w + 2)}┼{'─' * 12}┤"
    bot = f"  └{'─' * (label_w + 2)}┴{'─' * 12}┘"
    print(top)
    for i, (label, value) in enumerate(rows):
        if i == 1:
            print(mid)
        print(f"  │ {label:<{label_w}} │ {value:>10} │")
    print(bot)
    print()


def _print_leakage_table(report):
    """Pretty-print the cross-subject contamination probe."""
    print(f"  Cross-Subject Leakage Probe (subject filter DISABLED, k={report.n_items and 'top-k' or '-'})")
    print()
    rows = [
        ("On-subject rate", f"{report.on_subject_rate:.2%}"),
        ("Leakage rate", f"{report.leakage_rate:.2%}"),
        ("Items probed", str(report.n_items)),
    ]
    if report.n_errors:
        rows.append(("Probe errors", str(report.n_errors)))
    label_w = max(len(r[0]) for r in rows)
    top = f"  ┌{'─' * (label_w + 2)}┬{'─' * 12}┐"
    mid = f"  ├{'─' * (label_w + 2)}┼{'─' * 12}┤"
    bot = f"  └{'─' * (label_w + 2)}┴{'─' * 12}┘"
    print(top)
    for i, (label, value) in enumerate(rows):
        if i == 1:
            print(mid)
        print(f"  │ {label:<{label_w}} │ {value:>10} │")
    print(bot)
    if report.items:
        print("  Per-query retrieved subjects:")
        for it in report.items:
            if it.get("error"):
                print(f"    · {it['query']} -> ERROR: {it['error']}")
            else:
                subs = it.get("retrieved_subjects", [])
                print(f"    · {it['query']} -> {subs}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="RAG Tutor CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run UI command
    subparsers.add_parser("run", help="Start the Streamlit Web UI interface")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest study notes PDF files for a subject")
    ingest_parser.add_argument(
        "--subject", 
        default="Operating System",
        help="Subject name to ingest (default: Operating System)"
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run the RAG retrieval evaluation harness (Hit Rate, MRR, Precision/Recall@k)"
    )
    eval_parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a ground-truth dataset JSON (default: built-in dataset)"
    )
    eval_parser.add_argument(
        "--subject",
        default=None,
        help="Subject to evaluate against (default: first item's subject)"
    )
    eval_parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Retrieval depth and @k for precision/recall (default: 4)"
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a table"
    )
    eval_parser.add_argument(
        "--leakage",
        action="store_true",
        help="Also probe cross-subject contamination (retrieval with subject filter disabled)"
    )

    args = parser.parse_args()
    
    if args.command == "run":
        run_ui()
    elif args.command == "ingest":
        run_ingest(args.subject)
    elif args.command == "eval":
        run_eval(
            dataset_path=args.dataset,
            subject=args.subject,
            k=args.k,
            as_json=args.json,
            leakage=args.leakage,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
