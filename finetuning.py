"""
Complete Fine-tuning Script for Google Colab
Fine-tune Qwen 2.5 for KPI Tool Calling (Function Calling Only)

Instructions:
1. Upload your data.json to Colab
2. Run all cells in order
3. Model outputs JSON function calls only - no explanations
"""

# ============================================================================
# SECTION 1: INSTALLATION & SETUP
# ============================================================================

import subprocess
import sys

print("=" * 80)
print("STEP 1: Installing Required Packages")
print("=" * 80)

# Install required packages
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
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import List, Dict
import numpy as np

print("✓ Imports successful!")

# Check GPU with proper error handling
DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Clear CUDA cache to avoid memory issues
        torch.cuda.empty_cache()
    else:
        print("⚠ Warning: No GPU detected. Using CPU (training will be slower).")
except Exception as e:
    print(f"⚠ CUDA initialization error: {e}")
    print("  Falling back to CPU.")
    DEVICE = "cpu"

# ============================================================================
# SECTION 3: CONFIGURATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Configuration")
print("=" * 80)

# Training Configuration
CONFIG = {
    # Model settings
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    
    # Data settings
    "train_file": "data.json",  # Your training data file
    "test_split": 0.1,  # 10% for validation (keep more for training with small dataset)
    "max_length": 2048,
    
    # Training hyperparameters
    "num_epochs": 5,  # More epochs for small dataset
    "batch_size": 1,  # Smaller batch size = more steps = better logging
    "gradient_accumulation_steps": 4,  # Effective batch size of 4
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    
    # LoRA settings
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    
    # Output settings
    "output_dir": "./qwen-kpi-finetuned",
    "save_steps": 20,
    "logging_steps": 1,  # Log every step to see training progress
}

# Print configuration
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================================
# SECTION 4: KPI FUNCTION SCHEMA (get_oee tool)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Loading KPI Function Schema")
print("=" * 80)

# Tool schema matching the data.json format (get_oee)
kpi_function_schema = {
    "type": "function",
    "function": {
        "name": "get_oee",
        "description": "Use this tool when the user asks for OEE or overall effectiveness or availability metrics for specific lines, machines, areas, plants, or products over a defined time period.",
        "parameters": {
            "type": "object",
            "properties": {
                "custom_start_date": {
                    "type": "string",
                    "description": "The start date and time for the data query in 'YYYY-MM-DD HH:MM:SS' format.",
                },
                "custom_end_date": {
                    "type": "string",
                    "description": "The end date and time for the data query in 'YYYY-MM-DD HH:MM:SS' format.",
                },
                "daily": {"type": "boolean", "description": "Set to true for daily breakdown"},
                "hourly": {"type": "boolean", "description": "Set to true for hourly breakdown"},
                "monthly": {"type": "boolean", "description": "Set to true for monthly breakdown"},
                "weekly": {"type": "boolean", "description": "Set to true for weekly breakdown"},
                "yearly": {"type": "boolean", "description": "Set to true for yearly breakdown"},
                "quarterly": {"type": "boolean", "description": "Set to true for quarterly breakdown"},
                "highest": {"type": "boolean", "description": "Set to true for highest values"},
                "lowest": {"type": "boolean", "description": "Set to true for lowest values"},
                "exclude_zero": {"type": "boolean", "description": "Set to true to exclude zero values"},
                "highest_to_lowest": {"type": "boolean", "description": "Sort from highest to lowest"},
                "lowest_to_highest": {"type": "boolean", "description": "Sort from lowest to highest"},
                "top": {"type": "integer", "description": "Number of top records to return"},
                "plant": {"type": "string", "description": "Filter by plant name"},
                "area": {"type": "string", "description": "Filter by area name"},
                "machine": {"type": "string", "description": "Filter by machine name"},
                "line": {"type": "string", "description": "Filter by line name"},
                "all_plant": {"type": "boolean", "description": "Include all plants"},
                "all_area": {"type": "boolean", "description": "Include all areas"},
                "all_machine": {"type": "boolean", "description": "Include all machines"},
                "all_line": {"type": "boolean", "description": "Include all lines"},
                "highest_machine": {"type": "boolean", "description": "Get highest performing machine"},
                "lowest_machine": {"type": "boolean", "description": "Get lowest performing machine"},
                "rank": {"type": "boolean", "description": "Rank the results"},
                "shift": {"type": "string", "description": "Filter by shift name"},
                "all_shift": {"type": "boolean", "description": "Include all shifts"},
                "product": {"type": "string", "description": "Filter by product name"},
                "all_product": {"type": "boolean", "description": "Include all products"},
                "greater_than": {"type": "string", "description": "Filter OEE greater than value"},
                "less_than": {"type": "string", "description": "Filter OEE less than value"},
                "only_negative": {"type": "boolean", "description": "Only negative values"},
                "only_positive": {"type": "boolean", "description": "Only positive values"},
            },
            "required": ["custom_start_date", "custom_end_date"],
        },
    }
}

print("✓ KPI schema loaded (get_oee tool)")

# ============================================================================
# SECTION 5: DATA LOADING & VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Loading and Validating Training Data")
print("=" * 80)

def load_and_validate_data(file_path: str) -> List[Dict]:
    """Load and validate training data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} training examples")
    
    # Validate
    for i, sample in enumerate(data):
        assert "user_content" in sample, f"Sample {i}: missing 'user_content'"
        assert "tool_calls" in sample, f"Sample {i}: missing 'tool_calls'"
        
        for j, tool_call in enumerate(sample["tool_calls"]):
            assert "tool_name" in tool_call, f"Sample {i}, tool_call {j}: missing 'tool_name'"
            assert "tool_arguments" in tool_call, f"Sample {i}, tool_call {j}: missing 'tool_arguments'"
            
            # Parse and validate arguments
            args = tool_call["tool_arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            
            assert "custom_start_date" in args, f"Sample {i}: missing custom_start_date"
            assert "custom_end_date" in args, f"Sample {i}: missing custom_end_date"
    
    print("✓ All samples validated successfully!")
    return data

# Load data
raw_data = load_and_validate_data(CONFIG["train_file"])

# Show statistics
print(f"\nData Statistics:")
print(f"  Total examples: {len(raw_data)}")
print(f"  Average query length: {np.mean([len(s['user_content']) for s in raw_data]):.1f} characters")

# Show sample
print(f"\nSample query:")
print(f"  User: {raw_data[0]['user_content']}")
print(f"  Tool: {raw_data[0]['tool_calls'][0]['tool_name']}")
print(f"  Args: {json.dumps(raw_data[0]['tool_calls'][0]['tool_arguments'], indent=4)[:200]}...")

# ============================================================================
# SECTION 6: DATA FORMATTING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Formatting Data for Training")
print("=" * 80)

# System prompt for JSON-only function calling output
SYSTEM_PROMPT = """You are a function calling assistant. Your ONLY task is to analyze user requests about OEE (Overall Equipment Effectiveness) data and respond with the appropriate function call(s) in JSON format.

Rules:
1. ONLY output function calls - NO explanations, NO text responses
2. For comparison queries (e.g., comparing two lines or shifts), make SEPARATE function calls for each item being compared
3. Always include custom_start_date and custom_end_date in every function call
4. Output must be valid JSON function call format"""

def format_data_for_training(raw_data: List[Dict], tokenizer) -> Dataset:
    """Convert JSON data to Qwen tool calling format for function-calling only"""
    formatted_samples = []
    
    for sample in raw_data:
        user_content = sample['user_content']
        tool_calls = sample['tool_calls']
        
        # Create conversation with function-calling only system prompt
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        # Build all tool calls for this sample (supports multiple calls for comparisons)
        assistant_tool_calls = []
        for tool_call in tool_calls:
            tool_name = tool_call['tool_name']  # This will be 'get_oee' from data.json
            tool_args = tool_call['tool_arguments']
            
            # Parse if string
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
            
            assistant_tool_calls.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args)
                }
            })
        
        # Create single assistant message with all tool calls
        assistant_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": assistant_tool_calls
        }
        messages.append(assistant_message)
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tools=[kpi_function_schema],
            tokenize=False,
            add_generation_prompt=False
        )
        
        formatted_samples.append({"text": text})
    
    return Dataset.from_list(formatted_samples)

# ============================================================================
# SECTION 7: MODEL LOADING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Loading Model and Tokenizer")
print("=" * 80)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name"],
    trust_remote_code=True,
    padding_side="right"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Tokenizer loaded")

# Load model with proper error handling
print("Loading model (this may take a few minutes)...")
try:
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # CPU fallback - use float32 for CPU
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"],
            torch_dtype=torch.float32,
            device_map={"":"cpu"},
            trust_remote_code=True,
        )
    print("✓ Model loaded")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative loading method...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print("✓ Model loaded (CPU mode)")

# Configure LoRA
print("Configuring LoRA...")
peft_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training - important for gradient checkpointing compatibility
model.enable_input_require_grads()  # Required for gradient checkpointing with LoRA

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Ensure model is in training mode
model.train()

print("✓ LoRA configured")

# ============================================================================
# SECTION 8: DATA PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Preparing Training Data")
print("=" * 80)

# Format data
print("Formatting data...")
dataset = format_data_for_training(raw_data, tokenizer)
print(f"✓ Formatted {len(dataset)} examples")

# Tokenize - process one example at a time to avoid length mismatch issues
def tokenize_function(example):
    """Tokenize a single example"""
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding=False,
        return_tensors=None
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing data...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False,  # Process one at a time to avoid tensor length issues
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)
print("✓ Tokenization complete")

# Split data
print(f"Splitting data (test_size={CONFIG['test_split']})...")
train_test_split = tokenized_dataset.train_test_split(
    test_size=CONFIG["test_split"],
    seed=42
)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"✓ Train samples: {len(train_dataset)}")
print(f"✓ Eval samples: {len(eval_dataset)}")

# Custom data collator that properly handles padding for variable length sequences
from dataclasses import dataclass
from typing import Any, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling that handles variable length sequences.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate labels from features
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0] else None
        
        # Pad input_ids and attention_mask
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Handle labels - pad with -100 (ignored in loss computation)
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            padded_labels = []
            for label in labels:
                remainder = [-100] * (max_label_length - len(label))
                if padding_side == "right":
                    padded_labels.append(label + remainder)
                else:
                    padded_labels.append(remainder + label)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch

data_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    padding=True,
    max_length=CONFIG["max_length"],
)

# ============================================================================
# SECTION 9: TRAINING SETUP
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Setting Up Training")
print("=" * 80)

# Configure training arguments based on available hardware
use_fp16 = DEVICE == "cuda"

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=CONFIG["warmup_steps"],
    logging_steps=CONFIG["logging_steps"],
    logging_first_step=True,  # Log the first step
    save_steps=CONFIG["save_steps"],
    eval_strategy="epoch",  # Evaluate at end of each epoch
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="none",
    fp16=use_fp16,
    gradient_checkpointing=True,
    optim="adamw_torch" if DEVICE == "cpu" else "paged_adamw_8bit",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_nan_inf_filter=False,  # Don't filter nan/inf to see issues
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("✓ Trainer initialized")
print(f"\nTraining Configuration:")
print(f"  Total epochs: {CONFIG['num_epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
print(f"  Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Total steps: {len(train_dataset) // (CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']) * CONFIG['num_epochs']}")

# ============================================================================
# SECTION 10: TRAINING
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Starting Training")
print("=" * 80)
print("This will take some time. Monitor the loss - it should decrease steadily.")
print("=" * 80 + "\n")

# Start training
start_time = datetime.now()
trainer.train()
end_time = datetime.now()

print("\n" + "=" * 80)
print("✓ Training Complete!")
print("=" * 80)
print(f"Training time: {end_time - start_time}")

# ============================================================================
# SECTION 11: SAVE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Saving Final Model")
print("=" * 80)

final_model_path = os.path.join(CONFIG["output_dir"], "final_model")
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"✓ Model saved to: {final_model_path}")

# ============================================================================
# SECTION 12: TESTING FUNCTION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Testing Fine-tuned Model")
print("=" * 80)

def test_model(model_path: str, test_queries: List[str]):
    """Test the fine-tuned model with sample queries - outputs JSON function calls only"""
    
    print(f"Loading model from {model_path}...")
    test_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    try:
        if DEVICE == "cuda":
            test_model_loaded = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            test_model_loaded = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map={"":"cpu"}
            )
    except Exception as e:
        print(f"Loading with fallback due to: {e}")
        test_model_loaded = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    
    print("✓ Model loaded\n")
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"Query: {query}\n")
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        text = test_tokenizer.apply_chat_template(
            messages,
            tools=[kpi_function_schema],
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = test_tokenizer([text], return_tensors="pt").to(test_model_loaded.device)
        
        with torch.no_grad():
            generated_ids = test_model_loaded.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=test_tokenizer.pad_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = test_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Model Response:")
        print(response)
        print(f"\n{'='*80}")
        
        results.append({
            "query": query,
            "response": response
        })
    
    return results

# ============================================================================
# SECTION 13: RUN TESTS
# ============================================================================

# Define test queries - including comparison queries to test multiple function calls
test_queries = [
    "Show me daily OEE for all machines from 2024-08-01 00:00:00 to 2024-08-31 23:59:59",
    "Get me the top 5 performing lines from 2024-01-01 00:00:00 to 2024-12-31 23:59:59",
    "What's the monthly OEE breakdown for Plant_Austin from 2024-06-01 00:00:00 to 2024-08-31 23:59:59",
    "Compare the OEE of LINE_A01 vs LINE_B02 from 2024-07-01 00:00:00 to 2024-07-31 23:59:59",
    "Compare OEE between Day Shift and Night Shift for the date range 2024-09-22 18:03:54 to 2024-09-23 23:49:11",
]

print("\nRunning tests on fine-tuned model...")
test_results = test_model(final_model_path, test_queries)

# ============================================================================
# SECTION 14: EVALUATION SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING AND TESTING COMPLETE!")
print("=" * 80)

print(f"\nModel Information:")
print(f"  Model path: {final_model_path}")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Evaluation samples: {len(eval_dataset)}")
print(f"  Training time: {end_time - start_time}")

print(f"\nTest Results:")
print(f"  Total test queries: {len(test_queries)}")
print(f"  Results available in 'test_results' variable")

print("\n" + "=" * 80)
print("Next Steps:")
print("=" * 80)
print("1. Review the test results above")
print("2. Try your own queries using test_model() function")
print("3. Download the model if needed (it's in the output directory)")
print("4. If results are not satisfactory:")
print("   - Add more diverse training examples")
print("   - Increase num_epochs")
print("   - Adjust learning_rate")
print("=" * 80)

# ============================================================================
# SECTION 15: HELPER FUNCTIONS FOR ADDITIONAL TESTING
# ============================================================================

def quick_test(query: str):
    """Quick test function for a single query"""
    results = test_model(final_model_path, [query])
    return results[0]

def parse_tool_call(response: str) -> dict:
    """Helper to parse tool call from model response"""
    try:
        # Look for JSON in the response
        import re
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return None

print("\n✓ Helper functions defined:")
print("  - quick_test(query): Test a single query")
print("  - parse_tool_call(response): Extract tool arguments from response")

print("\nExample usage:")
print('  result = quick_test("Show me top 5 machines from 2024-01-01 to 2024-12-31")')
print('  args = parse_tool_call(result["response"])')

print("\n" + "="*80)
print("ALL DONE! Your model is ready to use. 🎉")
print("="*80)