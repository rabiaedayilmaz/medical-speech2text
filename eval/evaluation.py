import os
import json
import difflib
from pathlib import Path
from speech.transcribe import transcribe_and_refine, transcribe_audio

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from utils.log import logger


# paths
GROUND_TRUTH_DIR = Path("test/test_text_data")
AUDIO_DIR = Path("test/test_voice_data")
RESULTS_DIR = Path("results/eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# test file names
TEST_FILES = [
    "test_medikal_apandisit",
    "test_medikal_dis",
    "test_medikal_femur",
    "test_medikal_kolesistektomi",
    "test_medikal_lichtenstein",
    "test_medikal_lumpektomi",
    "test_medikal_tah"
]

def load_ground_truth(file_stem: str) -> str:
    """Loads the ground truth text for the given file stem."""
    path = GROUND_TRUTH_DIR / f"{file_stem}.txt"
    if not path.exists():
        logger.warning(f"Ground truth not found for {file_stem}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text: str) -> str:
    """Cleans the given text for better comparison."""
    return (
        text.lower()
            .replace("-", " ")
            .replace("\u2022", "")
            .replace("*", "")
            .replace("\n", " ")
            .replace(",", " ")
            .replace(".", " ")
            .replace(";", " ")
            .replace(":", " ")
            .strip()
    )

def evaluate_file(file_stem: str, is_transcribe_and_refine: bool = False):
    """Evaluates the given audio file with respect to the ground truth."""
    audio_path = AUDIO_DIR / f"{file_stem}.mp3"
    logger.info(f"Evaluating: {file_stem}")

    if is_transcribe_and_refine:
        pred = transcribe_and_refine(str(audio_path))
    else:
        pred = transcribe_audio(str(audio_path))

    ground_truth = load_ground_truth(file_stem)
    gt_clean = clean_text(ground_truth)
    pred_clean = clean_text(pred)

    # SequenceMatcher
    sequence_score = difflib.SequenceMatcher(None, gt_clean, pred_clean).ratio()

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([gt_clean.split()], pred_clean.split(), smoothing_function=smoothie)

    # ROUGE-L
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score_l = rouge.score(gt_clean, pred_clean)['rougeL'].fmeasure

    # BERTScore (using semantic embeddings)
    P, R, F1 = bert_score([pred_clean], [gt_clean], lang="en", rescale_with_baseline=True)
    bert_f1 = F1[0].item()

    # save detailed result as json
    result_path = RESULTS_DIR / f"{file_stem}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "file": file_stem,
            "similarity_score": round(sequence_score, 4),
            "bleu": round(bleu, 4),
            "rougeL": round(rouge_score_l, 4),
            "bert_score_f1": round(bert_f1, 4),
            "ground_truth": ground_truth,
            "refined_transcript": pred
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Scores — Sim: {sequence_score:.4f}, BLEU: {bleu:.4f}, ROUGE-L: {rouge_score_l:.4f}, BERT: {bert_f1:.4f}")

    return file_stem, {
        "similarity": round(sequence_score, 4),
        "bleu": round(bleu, 4),
        "rougeL": round(rouge_score_l, 4),
        "bert_score_f1": round(bert_f1, 4),
    }

def run_evaluation():
    all_results = {}
    for file in TEST_FILES:
        stem, scores = evaluate_file(file, is_transcribe_and_refine=False)
        all_results[stem] = scores

    # save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info("\nEvaluation summary saved to:", summary_path)

if __name__ == "__main__":
    run_evaluation()