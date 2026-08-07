"""
Build the hybrid SFT dataset from data/raw/* into data/processed/hybrid/.

Tasks combined (see README/summary.md for the reasoning behind each):
  - medqa              theory QA (USMLE-style), low weight, general grounding
  - medalign           real clinician instructions on real EHRs
  - pubmedqa           biomedical literature QA (gold/expert-labeled subset only)
  - bhc                Brief Hospital Course summarization (Stanford MIMIC-IV-Ext-BHC)
  - icd_coding         discharge note -> ICD diagnosis codes (top-200 most frequent)
  - radiology          Findings -> Impression generation
  - discharge_ie       discharge note -> structured JSON fields
  - direct             KG-grounded differential diagnosis reasoning (MIMIC-IV-Ext-DiReCT)
  - med_case_reasoning published clinical case reports -> diagnostic reasoning
  - cdm                MIMIC-IV-Ext-CDM (locally generated) abdominal pathology cases
"""
import csv
import json
import os
import pickle
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

csv.field_size_limit(2**31 - 1)

RAW = "data/raw"
OUT = "data/processed/hybrid"
SEED = 42

N_MEDQA = 3000
N_PUBMEDQA = 1000
N_BHC = 6000
N_ICD = 5000
N_RADIOLOGY = 3000
N_DISCHARGE_IE = 3000
N_MEDCASEREASONING_TRAIN = 6000
ICD_TOP_K = 200
TEST_HOLDOUT = 300  # per-task held-out slice for tasks that didn't already have one


def msg(system, user, assistant):
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ---------------------------------------------------------------- medqa ----
def load_medqa():
    rows = [json.loads(l) for l in open(f"{RAW}/medqa/medical_meadow_medqa.json", encoding="utf-8")]
    random.shuffle(rows)
    test_rows, pool = rows[:TEST_HOLDOUT], rows[TEST_HOLDOUT:]
    train_rows = random.sample(pool, min(N_MEDQA, len(pool)))

    def to_examples(rs):
        return [
            msg(
                "You are a highly skilled medical professional. Answer the medical question accurately.",
                f"{r['instruction']}\n\n{r['input']}",
                r["output"],
            )
            for r in rs
        ]

    return to_examples(train_rows), to_examples(test_rows)


# -------------------------------------------------------------- medalign ---
def extract_notes_from_xml(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
        notes = [e.text.strip() for e in root.findall(".//event[@type='note']") if e.text]
        return "\n\n---\n\n".join(notes)
    except Exception:
        return ""


def load_medalign(test_size=50):
    base = f"{RAW}/medalign/instructions/medalign_instructions_v1_3"
    ehrs_dir = f"{base}/ehrs"
    rows = []
    with open(f"{base}/clinician-instruction-responses.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            filename, question, answer = row.get("filename"), row.get("question"), row.get("clinician_response")
            if not filename or not answer:
                continue
            xml_path = os.path.join(ehrs_dir, filename)
            if not os.path.exists(xml_path):
                continue
            notes = extract_notes_from_xml(xml_path)
            if not notes:
                continue
            rows.append(
                msg(
                    "You are a highly skilled medical AI assistant. Analyze the clinical notes and follow the instruction precisely.",
                    f"Clinical Notes:\n{notes}\n\nInstruction:\n{question}",
                    answer,
                )
            )
    random.shuffle(rows)
    return rows[test_size:], rows[:test_size]


# -------------------------------------------------------------- pubmedqa ---
def load_pubmedqa(test_size=100):
    data = json.load(open(f"{RAW}/pubmedqa/ori_pqal.json", encoding="utf-8"))
    items = list(data.values())
    random.shuffle(items)
    items = items[:N_PUBMEDQA]
    out = [
        msg(
            "You are a medical research assistant. Answer the biomedical research question based on the "
            "abstract context, starting with yes/no/maybe, then justify.",
            f"Context:\n{' '.join(r.get('CONTEXTS', []))}\n\nQuestion: {r['QUESTION']}",
            f"{r['final_decision'].capitalize()}. {r['LONG_ANSWER']}",
        )
        for r in items
    ]
    return out[test_size:], out[:test_size]


# ------------------------------------------------------------------ bhc ----
def load_bhc():
    rows = list(csv.DictReader(open(f"{RAW}/mimic_iv_ext_bhc.csv", encoding="utf-8")))
    random.shuffle(rows)
    test_rows, pool = rows[:TEST_HOLDOUT], rows[TEST_HOLDOUT:]
    train_rows = random.sample(pool, min(N_BHC, len(pool)))

    def to_examples(rs):
        return [
            msg(
                "You are a medical AI assistant. Summarize the hospital admission into a Brief Hospital Course, "
                "as written by a physician.",
                r["input"].replace("summarize:\n", "", 1).strip(),
                r["target"].strip(),
            )
            for r in rs
        ]

    return to_examples(train_rows), to_examples(test_rows)


# ------------------------------------------------------------- icd_coding --
def load_icd_coding():
    diag = pd.read_csv(f"{RAW}/mimic_iv/hosp/diagnoses_icd.csv")
    diag = diag[~diag.icd_code.isna()]
    desc = pd.read_csv(f"{RAW}/mimic_iv/hosp/d_icd_diagnoses.csv")
    diag = diag.merge(desc[["icd_code", "icd_version", "long_title"]], on=["icd_code", "icd_version"], how="left")

    top_codes = set(diag["icd_code"].value_counts().head(ICD_TOP_K).index)
    diag_top = diag[diag["icd_code"].isin(top_codes)]

    by_hadm = defaultdict(set)
    for r in diag_top.itertuples():
        by_hadm[int(r.hadm_id)].add(r.long_title)

    target_n = N_ICD + TEST_HOLDOUT
    target_ids = set(random.sample(list(by_hadm.keys()), min(target_n * 3, len(by_hadm))))

    examples = []
    with open(f"{RAW}/mimic_iv_note/note/discharge.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                hadm_id = int(row["hadm_id"])
            except (ValueError, TypeError):
                continue
            if hadm_id not in target_ids:
                continue
            codes = sorted(by_hadm[hadm_id])
            examples.append(
                msg(
                    "You are a certified medical coder. Read the discharge summary and list the relevant "
                    "ICD diagnosis codes (as their descriptive titles).",
                    row["text"],
                    "\n".join(f"- {c}" for c in codes),
                )
            )
            if len(examples) >= target_n:
                break
    random.shuffle(examples)
    return examples[TEST_HOLDOUT:], examples[:TEST_HOLDOUT]


# ------------------------------------------------------------- radiology ---
FINDINGS_IMPRESSION_RE = re.compile(r"FINDINGS:\s*\n(.*?)\n\s*IMPRESSION:\s*\n(.*?)(?:\n\n|\Z)", re.S)


def load_radiology():
    out = []
    with open(f"{RAW}/mimic_iv_note/note/radiology.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = FINDINGS_IMPRESSION_RE.search(row["text"])
            if not m:
                continue
            findings, impression = m.group(1).strip(), m.group(2).strip()
            if len(findings) < 20 or len(impression) < 5:
                continue
            out.append(
                msg(
                    "You are a radiologist. Write the Impression section for this radiology report based on the Findings.",
                    f"FINDINGS:\n{findings}",
                    impression,
                )
            )
            if len(out) >= (N_RADIOLOGY + TEST_HOLDOUT) * 2:
                break
    out = random.sample(out, min(N_RADIOLOGY + TEST_HOLDOUT, len(out)))
    return out[TEST_HOLDOUT:], out[:TEST_HOLDOUT]


# ---------------------------------------------------------- discharge_ie ---
# The full set of standard MIMIC-IV discharge summary headers - used as split boundaries
# so a section's content never bleeds into the next (a bare "\n\n" isn't reliable: MIMIC
# notes sometimes separate headers with a stray "\n \n" instead).
ALL_DISCHARGE_HEADERS = [
    "Chief Complaint",
    "Major Surgical or Invasive Procedure",
    "History of Present Illness",
    "Past Medical History",
    "Social History",
    "Family History",
    "Physical Exam",
    "Pertinent Results",
    "Brief Hospital Course",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
]
IE_SECTIONS = {
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Discharge Diagnosis",
    "Discharge Medications",
    "Discharge Condition",
}
ALL_HEADERS_SPLIT_RE = re.compile(r"\n(" + "|".join(re.escape(s) for s in ALL_DISCHARGE_HEADERS) + r"):\n")


def split_discharge_sections(text):
    parts = ALL_HEADERS_SPLIT_RE.split(text)
    result = {}
    for i in range(1, len(parts) - 1, 2):
        header, content = parts[i], parts[i + 1]
        if header in IE_SECTIONS:
            result[header] = content.strip()
    return result


def load_discharge_ie():
    out = []
    with open(f"{RAW}/mimic_iv_note/note/discharge.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sections = split_discharge_sections(row["text"])
            if len(sections) < 4:
                continue
            out.append(
                msg(
                    "You are a medical AI assistant. Extract the following fields from the discharge summary as JSON.",
                    row["text"],
                    json.dumps(sections, ensure_ascii=False, indent=2),
                )
            )
            if len(out) >= (N_DISCHARGE_IE + TEST_HOLDOUT) * 2:
                break
    out = random.sample(out, min(N_DISCHARGE_IE + TEST_HOLDOUT, len(out)))
    return out[TEST_HOLDOUT:], out[:TEST_HOLDOUT]


# ------------------------------------------------------------------ direct -
# Reimplementation of the MIMIC-IV-Ext-DiReCT annotation format
# (Wang et al., "DiReCT: Diagnostic Reasoning for Clinical Notes via LLMs", NeurIPS 2024).
# Each non-"inputN" top-level key is a tree rooted at the final diagnosis; node keys are
# "content$Type_N" and nested dict values are that node's children (see the dataset README).
DIRECT_SECTIONS = {
    "input1": "Chief Complaint",
    "input2": "History of Present Illness",
    "input3": "Past Medical History",
    "input4": "Family History",
    "input5": "Physical Exam",
    "input6": "Pertinent Results",
}


def parse_direct_json(path):
    data = json.load(open(path, encoding="utf-8"))
    input_content = {k: v.replace("﻿", "") for k, v in data.items() if k.startswith("input")}
    nodes = []

    def walk(key, children, parent):
        content, type_ = key.rsplit("$", 1)
        type_ = type_.split("_")[0]
        idx = len(nodes)
        nodes.append({"content": content, "type": type_, "parent": parent})
        for child_key, child_val in children.items():
            walk(child_key, child_val, idx)
        return idx

    for key, val in data.items():
        if not key.startswith("input"):
            walk(key, val, None)

    # Multiple evidence branches can independently re-reach the same named stage (the
    # annotation is a tree, not a DAG) - dedupe by content while preserving order.
    raw_chain = reversed([n["content"] for n in nodes if n["type"] == "Intermedia"])
    chain = list(dict.fromkeys(raw_chain))

    evidence_by_stage = defaultdict(list)
    for n in nodes:
        # "InputN" types have no "_"-separated suffix (unlike "Cause_1"/"Intermedia_5"),
        # so they survive the split("_")[0] above unchanged - match by prefix, not equality.
        if not n["type"].startswith("Input") or n["parent"] is None:
            continue
        rationale = nodes[n["parent"]]
        stage = nodes[rationale["parent"]] if rationale["parent"] is not None else None
        if stage is None:
            continue
        evidence_by_stage[stage["content"]].append((n["content"], rationale["content"]))

    # Higher stages redundantly re-embed earlier stages' whole subtrees (see comment on
    # `chain` above), so the same (observation, rationale) pair can recur - dedupe per stage.
    evidence_by_stage = {k: list(dict.fromkeys(v)) for k, v in evidence_by_stage.items()}

    return input_content, chain, evidence_by_stage


def build_direct_reasoning(chain, evidence_by_stage):
    lines = []
    for stage in chain:
        ev = evidence_by_stage.get(stage, [])
        if ev:
            ev_text = "; ".join(f'"{o}" ({r})' for o, r in ev)
            lines.append(f"- Considering {stage}, supported by: {ev_text}")
        else:
            lines.append(f"- Progressing to {stage}")
    lines.append(f"\nFinal diagnosis: {chain[-1]}")
    return "\n".join(lines)


def load_direct(min_train_per_category=4, eval_fraction=0.15):
    finished_dir = f"{RAW}/mimic_iv_ext_direct/Finished"
    kg_dir = f"{RAW}/mimic_iv_ext_direct/diagnostic_kg/Diagnosis_flowchart"
    train, test = [], []
    for category in sorted(os.listdir(finished_dir)):
        cat_path = os.path.join(finished_dir, category)
        if not os.path.isdir(cat_path):
            continue
        kg_path = os.path.join(kg_dir, category + ".json")
        knowledge = ""
        if os.path.exists(kg_path):
            kg = json.load(open(kg_path, encoding="utf-8"))
            knowledge = json.dumps(kg.get("knowledge", {}), ensure_ascii=False, indent=2)

        files = []
        for pdd in os.listdir(cat_path):
            pdd_path = os.path.join(cat_path, pdd)
            if os.path.isdir(pdd_path):
                files.extend(os.path.join(pdd_path, f) for f in os.listdir(pdd_path) if f.endswith(".json"))
        random.shuffle(files)

        examples = []
        for fpath in files:
            try:
                input_content, chain, evidence = parse_direct_json(fpath)
            except Exception:
                continue
            if not chain:
                continue
            note = "\n\n".join(f"{DIRECT_SECTIONS[k]}:\n{v}" for k, v in input_content.items() if k in DIRECT_SECTIONS)
            reasoning = build_direct_reasoning(chain, evidence)
            user = (
                f"Diagnostic knowledge for {category}:\n{knowledge}\n\n"
                f"Patient note:\n{note}\n\n"
                "Provide a step-by-step differential diagnosis grounded in the knowledge above, ending with the final diagnosis."
            )
            examples.append(
                msg(
                    "You are a physician performing diagnostic reasoning. Ground every step in the provided clinical knowledge and note evidence.",
                    user,
                    reasoning,
                )
            )

        if len(examples) < min_train_per_category:
            test.extend(examples)
        else:
            n_test = max(1, int(len(examples) * eval_fraction))
            test.extend(examples[:n_test])
            train.extend(examples[n_test:])
    return train, test


# ------------------------------------------------------- med_case_reasoning
def load_med_case_reasoning():
    base = f"{RAW}/med_case_reasoning/data"
    train_df = pd.read_parquet(f"{base}/train-00000-of-00001.parquet")
    test_df = pd.read_parquet(f"{base}/test-00000-of-00001.parquet")
    if len(train_df) > N_MEDCASEREASONING_TRAIN:
        train_df = train_df.sample(N_MEDCASEREASONING_TRAIN, random_state=SEED)

    def to_examples(df):
        return [
            msg(
                "You are a physician performing diagnostic reasoning on a published clinical case.",
                f"Case presentation:\n{r.case_prompt}\n\nProvide your diagnostic reasoning and final diagnosis.",
                f"{r.diagnostic_reasoning}\n\nFinal diagnosis: {r.final_diagnosis}",
            )
            for r in df.itertuples()
        ]

    return to_examples(train_df), to_examples(test_df)


# -------------------------------------------------------------------- cdm --
def load_cdm(eval_fraction=0.1):
    lab_map = pd.read_csv(f"{RAW}/mimic_iv_ext_cdm/lab_test_mapping.csv")
    lab_labels = dict(zip(lab_map["itemid"], lab_map["label"]))

    train, test = [], []
    for patho in ["appendicitis", "cholecystitis", "pancreatitis", "diverticulitis"]:
        cases = pickle.load(open(f"{RAW}/mimic_iv_ext_cdm/{patho}_hadm_info_first_diag.pkl", "rb"))
        items = list(cases.items())
        random.shuffle(items)
        n_test = max(1, int(len(items) * eval_fraction))
        for i, (hadm_id, case) in enumerate(items):
            labs_text = "\n".join(
                f"{lab_labels.get(int(itemid), itemid)}: {value}" for itemid, value in case.get("Laboratory Tests", {}).items()
            )
            rad_text = "\n\n".join(r.get("Report", "") for r in case.get("Radiology", []))
            user = (
                f"History of Present Illness:\n{case.get('Patient History', '')}\n\n"
                f"Physical Examination:\n{case.get('Physical Examination', '')}\n\n"
                f"Laboratory Tests:\n{labs_text}\n\n"
                f"Radiology:\n{rad_text}\n\n"
                "What is the most likely diagnosis?"
            )
            example = msg(
                "You are a physician in the emergency department evaluating a patient with abdominal pain.",
                user,
                f"Diagnosis: {case.get('Discharge Diagnosis', patho)}",
            )
            (test if i < n_test else train).append(example)
    return train, test


def main():
    random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    tasks = {
        "medqa": load_medqa,
        "medalign": load_medalign,
        "pubmedqa": load_pubmedqa,
        "bhc": load_bhc,
        "icd_coding": load_icd_coding,
        "radiology": load_radiology,
        "discharge_ie": load_discharge_ie,
        "direct": load_direct,
        "med_case_reasoning": load_med_case_reasoning,
        "cdm": load_cdm,
    }

    train_all, test_all = [], []
    for name, loader in tasks.items():
        print(f"Loading {name}...")
        train, test = loader()
        print(f"  {name}: {len(train)} train, {len(test)} eval")
        train_all.extend(train)
        for ex in test:
            ex["task"] = name
            test_all.append(ex)

    random.shuffle(train_all)
    val_size = int(len(train_all) * 0.05)
    val, train = train_all[:val_size], train_all[val_size:]

    for fname, rows in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test_all)]:
        with open(f"{OUT}/{fname}", "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nHybrid dataset ready. Train: {len(train)}, Val: {len(val)}, Test: {len(test_all)}")
    print(f"Saved to {OUT}/")


if __name__ == "__main__":
    main()
