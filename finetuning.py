"""
Complete Fine-tuning Script for Google Colab
Fine-tune Qwen 2.5 for KPI Tool Calling (Function Calling Only)

Tools:
- get_oee: For OEE/Overall Equipment Effectiveness queries
- get_availability: For availability/uptime queries

Evaluation:
- Pre-training evaluation on test set (baseline)
- Post-training evaluation on test set
- Scoring: Correct, Partially Correct, Incorrect

Data Files:
- training_data.json: Split into train/validation
- testing_data.json: Held out for final evaluation
"""

# ============================================================================
# SECTION 1: INSTALLATION & SETUP
# ============================================================================

import subprocess
import sys

print("=" * 80)
print("STEP 1: Installing Required Packages")
print("=" * 80)

packages = [
    "transformers>=4.40.0",
    "datasets>=2.16.0",
    "peft>=0.7.1",
    "accelerate>=0.25.0",
    "bitsandbytes>=0.41.3",
    "sentencepiece",
    "trl>=0.7.0"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✓ Installation complete!")

# ============================================================================
# SECTION 2: IMPORTS
# ============================================================================

import json
import torch
import os
import re
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from typing import List, Dict, Any, Tuple
import numpy as np

print("✓ Imports successful!")

# Check GPU
DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    else:
        print("⚠ Warning: No GPU detected. Using CPU.")
except Exception as e:
    print(f"⚠ CUDA error: {e}. Using CPU.")
    DEVICE = "cpu"

# ============================================================================
# SECTION 3: CONFIGURATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Configuration")
print("=" * 80)

CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "train_file": "training_data.json",
    "test_file": "testing_data.json",
    # Data: 500 samples → 450 train, 50 validation (10% split)
    "validation_split": 0.1,
    "max_length": 2048,
    # Epochs: 3-5 is optimal for 500 samples (avoids overfitting)
    "num_epochs": 4,
    # Effective batch size = batch_size × gradient_accumulation = 2×8 = 16
    # Good for stable training with limited GPU memory
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    # Learning rate: 1e-4 to 2e-4 works well for LoRA fine-tuning
    "learning_rate": 1.5e-4,
    # Warmup: ~10% of total steps (450 samples / 16 effective batch × 4 epochs ≈ 112 steps)
    "warmup_steps": 10,
    # LoRA: r=32 gives more capacity for learning tool patterns
    "lora_r": 32,
    "lora_alpha": 64,  # alpha = 2×r is common
    "lora_dropout": 0.1,  # Slightly higher dropout to prevent overfitting
    "output_dir": "./qwen-kpi-finetuned",
    # Log every 10 steps for cleaner output
    "logging_steps": 10,
    # Hugging Face Hub settings
    "push_to_hub": True,
    "hub_model_id": "your-username/qwen-kpi-tool-calling",  # Change to your HF username/repo
    "hub_token": None,  # Will be set from environment or input
}

for key, value in CONFIG.items():
    if key != "hub_token":  # Don't print token
        print(f"  {key}: {value}")

# ============================================================================
# SECTION 3.1: HUGGING FACE HUB SETUP
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2.1: Hugging Face Hub Setup")
print("=" * 80)

if CONFIG["push_to_hub"]:
    try:
        from huggingface_hub import login, HfApi
        
        # Try to get token from environment variable first
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        if not hf_token:
            # Prompt for token if not in environment
            print("\n⚠ No HF_TOKEN found in environment.")
            print("  Option 1: Set environment variable HF_TOKEN")
            print("  Option 2: Enter token below (get from https://huggingface.co/settings/tokens)")
            hf_token = input("\nEnter your Hugging Face token (or press Enter to skip): ").strip()
        
        if hf_token:
            CONFIG["hub_token"] = hf_token
            login(token=hf_token)
            print("✓ Logged in to Hugging Face Hub")
            
            # Verify token and get username
            api = HfApi()
            user_info = api.whoami(token=hf_token)
            hf_username = user_info["name"]
            print(f"  Username: {hf_username}")
            
            # Update hub_model_id with actual username if using placeholder
            if "your-username" in CONFIG["hub_model_id"]:
                CONFIG["hub_model_id"] = f"{hf_username}/qwen-kpi-tool-calling"
                print(f"  Repository: {CONFIG['hub_model_id']}")
        else:
            print("⚠ No token provided. Model will NOT be pushed to Hub.")
            CONFIG["push_to_hub"] = False
    except Exception as e:
        print(f"⚠ Hugging Face Hub setup failed: {e}")
        print("  Model will NOT be pushed to Hub.")
        CONFIG["push_to_hub"] = False
else:
    print("Push to Hub: Disabled")

# ============================================================================
# SECTION 4: TOOL SCHEMAS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Loading Tool Schemas")
print("=" * 80)

common_properties = {
    "custom_start_date": {"type": "string", "description": "Start date in 'YYYY-MM-DD HH:MM:SS' format"},
    "custom_end_date": {"type": "string", "description": "End date in 'YYYY-MM-DD HH:MM:SS' format"},
    "customer_id": {"type": "string", "description": "Customer ID (default: acme-beverage-co)"},
    "daily": {"type": "boolean", "description": "Daily breakdown"},
    "hourly": {"type": "boolean", "description": "Hourly breakdown"},
    "monthly": {"type": "boolean", "description": "Monthly breakdown"},
    "weekly": {"type": "boolean", "description": "Weekly breakdown"},
    "yearly": {"type": "boolean", "description": "Yearly/annual breakdown"},
    "quarterly": {"type": "boolean", "description": "Quarterly breakdown"},
    "highest": {"type": "boolean", "description": "Highest values"},
    "lowest": {"type": "boolean", "description": "Lowest values"},
    "exclude_zero": {"type": "boolean", "description": "Exclude zero values"},
    "highest_to_lowest": {"type": "boolean", "description": "Sort highest to lowest"},
    "lowest_to_highest": {"type": "boolean", "description": "Sort lowest to highest"},
    "top": {"type": "integer", "description": "Number of top/bottom records"},
    "plant": {"type": "string", "description": "Filter by plant (e.g., Plant_Austin)"},
    "area": {"type": "string", "description": "Filter by area (e.g., Processing_Zone)"},
    "machine": {"type": "string", "description": "Filter by machine (e.g., LINE_A01_FILLER_M01)"},
    "line": {"type": "string", "description": "Filter by line (e.g., LINE_A01)"},
    "all_plant": {"type": "boolean", "description": "Include all plants"},
    "all_area": {"type": "boolean", "description": "Include all areas"},
    "all_machine": {"type": "boolean", "description": "Include all machines"},
    "all_line": {"type": "boolean", "description": "Include all lines"},
    "highest_machine": {"type": "boolean", "description": "Get highest performing machines"},
    "lowest_machine": {"type": "boolean", "description": "Get lowest performing machines"},
    "rank": {"type": "boolean", "description": "Rank results"},
    "shift": {"type": "string", "description": "Filter by shift (Day Shift, Night Shift, etc.)"},
    "all_shift": {"type": "boolean", "description": "Include all shifts"},
    "product": {"type": "string", "description": "Filter by product (e.g., Cola_330ml)"},
    "all_product": {"type": "boolean", "description": "Include all products"},
    "greater_than": {"type": "string", "description": "Filter > threshold"},
    "less_than": {"type": "string", "description": "Filter < threshold"},
}

get_oee_schema = {
    "type": "function",
    "function": {
        "name": "get_oee",
        "description": "Get OEE (Overall Equipment Effectiveness), equipment efficiency, or overall effectiveness metrics.",
        "parameters": {
            "type": "object",
            "properties": common_properties,
            "required": ["custom_start_date", "custom_end_date"],
        },
    }
}

get_availability_schema = {
    "type": "function",
    "function": {
        "name": "get_availability",
        "description": "Get availability, uptime, or machine availability metrics.",
        "parameters": {
            "type": "object",
            "properties": common_properties,
            "required": ["custom_start_date", "custom_end_date"],
        },
    }
}

TOOL_SCHEMAS = [get_oee_schema, get_availability_schema]

print("✓ Tool schemas loaded:")
print("  - get_oee: OEE/efficiency queries")
print("  - get_availability: availability/uptime queries")

# ============================================================================
# SECTION 5: SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a function calling assistant for manufacturing KPI data. Respond ONLY with function calls in JSON format - NO explanations.

Tools:
1. get_oee - For OEE, equipment efficiency, overall effectiveness
2. get_availability - For availability, uptime, machine availability

Rules:
1. Output ONLY function calls - NO text explanations
2. For comparisons, make SEPARATE calls for each item
3. If user asks for BOTH OEE and availability, call BOTH tools
4. Always include customer_id: "acme-beverage-co"
5. Always include custom_start_date and custom_end_date
6. Choose tool by keywords:
   - OEE/effectiveness/efficiency → get_oee
   - availability/uptime → get_availability
   - Both mentioned → call both tools"""

# ============================================================================
# SECTION 6: DATA LOADING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Loading Data")
print("=" * 80)

def load_data(file_path: str) -> List[Dict]:
    """Load and filter valid data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [item for item in data if item and "user_content" in item and "tool_calls" in item and item["tool_calls"]]

train_data = load_data(CONFIG["train_file"])
test_data = load_data(CONFIG["test_file"])

print(f"✓ Training examples: {len(train_data)}")
print(f"✓ Testing examples: {len(test_data)}")

# Analyze data
def analyze_data(data: List[Dict], name: str):
    oee_calls = sum(1 for item in data for tc in item.get("tool_calls", []) if tc.get("tool_name") == "get_oee")
    avail_calls = sum(1 for item in data for tc in item.get("tool_calls", []) if tc.get("tool_name") == "get_availability")
    multi_tool = sum(1 for item in data if len(item.get("tool_calls", [])) > 1)
    print(f"\n{name}: {len(data)} queries, {oee_calls} get_oee, {avail_calls} get_availability, {multi_tool} multi-tool")

analyze_data(train_data, "Training")
analyze_data(test_data, "Testing")

# ============================================================================
# SECTION 7: EVALUATION FUNCTIONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Evaluation Functions")
print("=" * 80)

def normalize_value(v):
    """Normalize a value for comparison - handle dates, booleans, nulls, empty strings"""
    if v is None or v == "" or v == "null":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        # Normalize date format: "2024-05-22T00:00:00" -> "2024-05-22 00:00:00"
        v = v.strip().replace("T", " ")
        # Handle string booleans
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        if v.lower() == "null" or v.lower() == "none":
            return None
    return v

def normalize_args(args: Dict) -> Dict:
    """Normalize arguments for comparison - only keep non-default values"""
    normalized = {}
    for k, v in args.items():
        norm_v = normalize_value(v)
        # Skip default/empty values (false, null, empty string, 0 for non-required fields)
        if norm_v is None or norm_v == False or norm_v == "":
            continue
        # Keep the normalized value
        normalized[k] = str(norm_v).strip() if isinstance(norm_v, str) else norm_v
    return normalized

def compare_tool_calls(predicted: List[Dict], expected: List[Dict]) -> Tuple[str, str]:
    """
    Compare predicted vs expected tool calls.
    Returns: (score, details)
    - CORRECT: All required tools called with correct required arguments
    - PARTIALLY_CORRECT: Some tools correct, or missing some tool calls
    - INCORRECT: Wrong tools or critical arguments wrong
    
    Rules:
    - Default values (false, null, empty) in predictions are OK
    - Date format differences (T vs space) are normalized
    - Focus on REQUIRED params: custom_start_date, custom_end_date, and filter params user specified
    - Extra default params in response don't affect scoring
    """
    if not predicted:
        return "INCORRECT", "No predictions"
    if not expected:
        return "INCORRECT", "No expected calls"
    
    # Normalize predictions
    pred_tools = []
    for p in predicted:
        name = p.get("tool_name") or p.get("function", {}).get("name") or p.get("name", "")
        args = p.get("tool_arguments") or p.get("function", {}).get("arguments") or p.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {}
        pred_tools.append({"name": name, "args": normalize_args(args)})
    
    # Normalize expected
    exp_tools = []
    for e in expected:
        name = e.get("tool_name", "")
        args = e.get("tool_arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {}
        exp_tools.append({"name": name, "args": normalize_args(args)})
    
    # Match predictions to expected (greedy matching)
    matched_full = 0
    matched_partial = 0
    exp_matched = [False] * len(exp_tools)
    
    for pred in pred_tools:
        best_score = 0
        best_idx = -1
        best_is_full = False
        
        for idx, exp in enumerate(exp_tools):
            if exp_matched[idx]:
                continue
            if pred["name"] != exp["name"]:
                continue
            
            # Compare arguments - check if prediction has all expected non-default args
            exp_args = exp["args"]
            pred_args = pred["args"]
            
            if not exp_args:
                # No specific args expected, tool name match is enough
                score = 1.0
                is_full = True
            else:
                # Count how many expected args are correctly matched
                matched_args = 0
                total_expected = len(exp_args)
                
                for k, v in exp_args.items():
                    pred_v = pred_args.get(k)
                    if pred_v is not None:
                        # Normalize both for comparison
                        exp_normalized = str(normalize_value(v)).strip() if v else None
                        pred_normalized = str(normalize_value(pred_v)).strip() if pred_v else None
                        if exp_normalized == pred_normalized:
                            matched_args += 1
                        elif k == "customer_id":
                            # customer_id is always the same, don't penalize
                            matched_args += 1
                
                score = matched_args / total_expected if total_expected > 0 else 1.0
                is_full = (matched_args == total_expected)
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_is_full = is_full
        
        if best_idx >= 0 and best_score >= 0.5:
            exp_matched[best_idx] = True
            if best_is_full or best_score >= 0.95:
                matched_full += 1
            else:
                matched_partial += 1
    
    # Calculate final score
    total_expected = len(exp_tools)
    total_matched = sum(exp_matched)
    
    # Determine overall score
    if matched_full == total_expected and len(pred_tools) >= total_expected:
        return "CORRECT", f"All {total_expected} tool calls correct"
    elif matched_full > 0 or matched_partial > 0:
        unmatched = total_expected - total_matched
        details = f"{matched_full} exact, {matched_partial} partial matches"
        if unmatched > 0:
            details += f", {unmatched} missing"
        if len(pred_tools) < total_expected:
            details += f" (predicted {len(pred_tools)}, expected {total_expected})"
        return "PARTIALLY_CORRECT", details
    else:
        pred_names = [p["name"] for p in pred_tools]
        exp_names = [e["name"] for e in exp_tools]
        return "INCORRECT", f"Expected {exp_names}, got {pred_names}"

def parse_model_response(response: str) -> List[Dict]:
    """Parse model response to extract tool calls"""
    tool_calls = []
    response = response.strip()
    
    # Try JSON parsing
    try:
        if response.startswith('['):
            return json.loads(response)
        if response.startswith('{'):
            return [json.loads(response)]
    except:
        pass
    
    # Extract JSON objects
    brace_depth = 0
    start = -1
    for i, c in enumerate(response):
        if c == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                try:
                    obj = json.loads(response[start:i+1])
                    if obj.get("name") in ["get_oee", "get_availability"]:
                        tool_calls.append({
                            "tool_name": obj["name"],
                            "tool_arguments": obj.get("arguments", {})
                        })
                    elif obj.get("tool_name") in ["get_oee", "get_availability"]:
                        tool_calls.append(obj)
                except:
                    pass
                start = -1
    
    return tool_calls

def evaluate_model(model, tokenizer, test_data: List[Dict], desc: str = "Evaluation") -> Dict:
    """Evaluate model on test data"""
    print(f"\n{'='*80}")
    print(f"{desc}")
    print(f"{'='*80}")
    
    results = {"correct": 0, "partially_correct": 0, "incorrect": 0, "total": len(test_data), "details": []}
    model.eval()
    
    for i, sample in enumerate(test_data):
        query = sample["user_content"]
        expected = sample["tool_calls"]
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        
        text = tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted = parse_model_response(response)
        score, details = compare_tool_calls(predicted, expected)
        
        results["details"].append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "response_raw": response,
            "score": score,
            "details": details
        })
        
        if score == "CORRECT":
            results["correct"] += 1
            score_symbol = "✓"
        elif score == "PARTIALLY_CORRECT":
            results["partially_correct"] += 1
            score_symbol = "~"
        else:
            results["incorrect"] += 1
            score_symbol = "✗"
        
        # Log each question with model response
        print(f"\n--- [{i+1}/{len(test_data)}] {score_symbol} {score} ---")
        print(f"  Query: {query}")
        print(f"  Expected Tools: {[tc['tool_name'] for tc in expected]}")
        pred_names = [tc.get('tool_name') or tc.get('name', '?') for tc in predicted]
        print(f"  Predicted Tools: {pred_names}")
        print(f"  Model Response: {response}")
        print(f"  Details: {details}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"{'='*80}")
    print(f"  ✓ Correct: {results['correct']} ({100*results['correct']/results['total']:.1f}%)")
    print(f"  ~ Partial: {results['partially_correct']} ({100*results['partially_correct']/results['total']:.1f}%)")
    print(f"  ✗ Incorrect: {results['incorrect']} ({100*results['incorrect']/results['total']:.1f}%)")
    
    return results

print("✓ Evaluation functions defined")

# ============================================================================
# SECTION 8: MODEL LOADING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Loading Model")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

if DEVICE == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"], torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"], torch_dtype=torch.float32, device_map={"": "cpu"}, trust_remote_code=True
    )
print("✓ Model loaded")

# ============================================================================
# SECTION 9: PRE-TRAINING EVALUATION (BASELINE)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Pre-Training Evaluation (Baseline)")
print("=" * 80)

pre_eval_size = min(20, len(test_data))
print(f"Evaluating base model on {pre_eval_size} test samples...")
pre_results = evaluate_model(model, tokenizer, test_data[:pre_eval_size], "PRE-TRAINING (Base Model)")

# ============================================================================
# SECTION 10: CONFIGURE LORA
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Configuring LoRA")
print("=" * 80)

peft_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)

model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.train()
print("✓ LoRA configured")

# ============================================================================
# SECTION 11: PREPARE DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Preparing Datasets")
print("=" * 80)

def format_for_training(data: List[Dict], tokenizer) -> Dataset:
    formatted = []
    for sample in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["user_content"]}
        ]
        
        tool_calls = [{
            "type": "function",
            "function": {"name": tc["tool_name"], "arguments": json.dumps(tc["tool_arguments"])}
        } for tc in sample["tool_calls"]]
        
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        text = tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=False)
        formatted.append({"text": text})
    
    return Dataset.from_list(formatted)

print("Formatting data...")
dataset = format_for_training(train_data, tokenizer)

def tokenize_fn(example):
    tok = tokenizer(example["text"], truncation=True, max_length=CONFIG["max_length"], padding=False, return_tensors=None)
    tok["labels"] = tok["input_ids"].copy()
    return tok

print("Tokenizing...")
tokenized = dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)

print("Splitting...")
split = tokenized.train_test_split(test_size=CONFIG["validation_split"], seed=42)
train_ds = split["train"]
val_ds = split["test"]
print(f"✓ Train: {len(train_ds)}, Validation: {len(val_ds)}")

# ============================================================================
# SECTION 12: DATA COLLATOR
# ============================================================================

from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features] if "labels" in features[0] else None
        batch = self.tokenizer.pad(features, padding=self.padding, max_length=self.max_length, return_tensors="pt")
        if labels:
            max_len = max(len(l) for l in labels)
            batch["labels"] = torch.tensor([l + [-100]*(max_len-len(l)) for l in labels], dtype=torch.long)
        return batch

data_collator = DataCollatorForCausalLM(tokenizer=tokenizer, padding=True, max_length=CONFIG["max_length"])

# ============================================================================
# SECTION 13: TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Training")
print("=" * 80)

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=CONFIG["warmup_steps"],
    logging_steps=CONFIG["logging_steps"],
    logging_first_step=True,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
    fp16=(DEVICE == "cuda"),
    gradient_checkpointing=True,
    optim="adamw_torch" if DEVICE == "cpu" else "paged_adamw_8bit",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

print(f"✓ Training: {CONFIG['num_epochs']} epochs, batch {CONFIG['batch_size']}x{CONFIG['gradient_accumulation_steps']}")

start_time = datetime.now()
trainer.train()
end_time = datetime.now()
print(f"\n✓ Training complete! Time: {end_time - start_time}")

# ============================================================================
# SECTION 14: SAVE MODEL (Local)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Saving Model Locally")
print("=" * 80)

final_path = os.path.join(CONFIG["output_dir"], "final_model")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"✓ Saved locally to: {final_path}")

# ============================================================================
# SECTION 15: POST-TRAINING EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 12: Post-Training Evaluation (Full Test Set)")
print("=" * 80)

print(f"Evaluating fine-tuned model on all {len(test_data)} test samples...")
post_results = evaluate_model(model, tokenizer, test_data, "POST-TRAINING (Fine-tuned)")

# ============================================================================
# SECTION 15.1: PUSH TO HUGGING FACE HUB (After Evaluation)
# ============================================================================

if CONFIG["push_to_hub"] and CONFIG["hub_token"]:
    print("\n" + "=" * 80)
    print("STEP 12.1: Pushing Model to Hugging Face Hub")
    print("=" * 80)
    
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        
        hub_model_id = CONFIG["hub_model_id"]
        print(f"Pushing to: https://huggingface.co/{hub_model_id}")
        
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            create_repo(
                repo_id=hub_model_id,
                token=CONFIG["hub_token"],
                private=False,  # Set to True for private repo
                exist_ok=True
            )
            print(f"✓ Repository ready: {hub_model_id}")
        except Exception as e:
            print(f"  Repository exists or created: {e}")
        
        # Create model card with actual evaluation results FIRST
        model_card = f"""---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- function-calling
- tool-use
- manufacturing
- kpi
- qwen
- peft
- lora
---

# Qwen2.5-1.5B Fine-tuned for KPI Tool Calling

This model is fine-tuned from [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) for manufacturing KPI tool calling.

## Tools

- **get_oee**: Get OEE (Overall Equipment Effectiveness) metrics
- **get_availability**: Get availability/uptime metrics

## Training Details

- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Fine-tuning Method**: LoRA (r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})
- **Training Samples**: {len(train_data)}
- **Epochs**: {CONFIG['num_epochs']}
- **Learning Rate**: {CONFIG['learning_rate']}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "{hub_model_id}")
tokenizer = AutoTokenizer.from_pretrained("{hub_model_id}")
```

## Evaluation Results

- **Correct**: {post_results['correct']}/{post_results['total']} ({100*post_results['correct']/post_results['total']:.1f}%)
- **Partially Correct**: {post_results['partially_correct']}/{post_results['total']} ({100*post_results['partially_correct']/post_results['total']:.1f}%)
- **Incorrect**: {post_results['incorrect']}/{post_results['total']} ({100*post_results['incorrect']/post_results['total']:.1f}%)
"""
        
        # Save model card to the model folder
        readme_path = os.path.join(final_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card)
        print(f"✓ Model card created")
        
        # Upload entire model folder using upload_folder
        print("Uploading model files...")
        upload_folder(
            folder_path=final_path,
            repo_id=hub_model_id,
            token=CONFIG["hub_token"],
            commit_message="Fine-tuned Qwen2.5-1.5B for KPI tool calling (get_oee, get_availability)"
        )
        print(f"✓ Model uploaded to Hub!")
        
        # Also push tokenizer explicitly (in case it wasn't in final_path)
        tokenizer.push_to_hub(
            repo_id=hub_model_id,
            token=CONFIG["hub_token"]
        )
        print(f"✓ Tokenizer pushed to Hub!")
        
        print(f"\n🎉 Model available at: https://huggingface.co/{hub_model_id}")
        
    except Exception as e:
        print(f"✗ Failed to push to Hub: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠ Push to Hub: Skipped (no token or disabled)")

# ============================================================================
# SECTION 16: RESULTS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION COMPARISON")
print("=" * 80)

print(f"\nPRE-TRAINING ({pre_eval_size} samples):")
print(f"  ✓ Correct: {pre_results['correct']} ({100*pre_results['correct']/pre_results['total']:.1f}%)")
print(f"  ~ Partial: {pre_results['partially_correct']} ({100*pre_results['partially_correct']/pre_results['total']:.1f}%)")
print(f"  ✗ Incorrect: {pre_results['incorrect']} ({100*pre_results['incorrect']/pre_results['total']:.1f}%)")

print(f"\nPOST-TRAINING ({len(test_data)} samples):")
print(f"  ✓ Correct: {post_results['correct']} ({100*post_results['correct']/post_results['total']:.1f}%)")
print(f"  ~ Partial: {post_results['partially_correct']} ({100*post_results['partially_correct']/post_results['total']:.1f}%)")
print(f"  ✗ Incorrect: {post_results['incorrect']} ({100*post_results['incorrect']/post_results['total']:.1f}%)")

# Save results
results_log = {
    "timestamp": datetime.now().isoformat(),
    "config": CONFIG,
    "training_time": str(end_time - start_time),
    "pre_training": {
        "samples": pre_eval_size,
        "correct": pre_results["correct"],
        "partial": pre_results["partially_correct"],
        "incorrect": pre_results["incorrect"],
        "correct_pct": 100*pre_results["correct"]/pre_results["total"],
        "details": pre_results["details"]
    },
    "post_training": {
        "samples": len(test_data),
        "correct": post_results["correct"],
        "partial": post_results["partially_correct"],
        "incorrect": post_results["incorrect"],
        "correct_pct": 100*post_results["correct"]/post_results["total"],
        "details": post_results["details"]
    }
}

results_file = os.path.join(CONFIG["output_dir"], "evaluation_results.json")
with open(results_file, "w") as f:
    json.dump(results_log, f, indent=2, default=str)
print(f"\n✓ Results saved to: {results_file}")

# ============================================================================
# SECTION 17: SAMPLE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAMPLE TEST RESULTS (First 5)")
print("=" * 80)

for i, r in enumerate(post_results["details"][:5]):
    print(f"\n--- Test {i+1} ---")
    print(f"Query: {r['query'][:80]}...")
    print(f"Score: {r['score']}")
    print(f"Expected: {[tc['tool_name'] for tc in r['expected']]}")
    pred_names = [tc.get('tool_name') or tc.get('name', '?') for tc in r['predicted']]
    print(f"Predicted: {pred_names}")
    print(f"Details: {r['details']}")

print("\n" + "="*80)
print("ALL DONE! 🎉")
print("="*80)
print(f"\nModel: {final_path}")
print(f"Results: {results_file}")
