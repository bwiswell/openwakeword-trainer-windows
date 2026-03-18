import glob
import os
import shutil
import sys
import subprocess
import yaml
import torchaudio
import openwakeword
import openwakeword.utils
from pathlib import Path
import runpy
import scipy.special
import torch
import types

# ==========================================
# 1. THE MONKEY PATCHES
# ==========================================
# These must run BEFORE importing openwakeword or speechbrain

OWW_RES_DIR = "H:/flora_data/openwakeword_resources"
MELSPEC_MODEL = os.path.join(OWW_RES_DIR, "melspectrogram.onnx")
EMBEDDING_MODEL = os.path.join(OWW_RES_DIR, "embedding_model.onnx")

# 2. THE MONKEY PATCH: Override the AudioFeatures constructor
original_init = openwakeword.utils.AudioFeatures.__init__

def patched_init(self, *args, **kwargs):
    # Force the model paths to our H: drive locations
    kwargs['melspec_model_path'] = MELSPEC_MODEL
    kwargs['embedding_model_path'] = EMBEDDING_MODEL
    return original_init(self, *args, **kwargs)

if not hasattr(scipy.special, "sph_harm"):
    print("Patching scipy.special.sph_harm for acoustics library compatibility...")
    scipy.special.sph_harm = scipy.special.sph_harm_y

torch.serialization.add_safe_globals(["piper_train.vits.models.SynthesizerTrn"])
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


if not hasattr(torchaudio, "list_audio_backends"):
    def list_audio_backends():
        return ["soundfile"]
    torchaudio.list_audio_backends = list_audio_backends

import openwakeword.data

# Inject our patched version into the library
openwakeword.utils.AudioFeatures.__init__ = patched_init
print(f"✅ Deep Hijack: AudioFeatures now hardcoded to use models in {OWW_RES_DIR}")

openwakeword.data.DEFAULT_BACKGROUND_PATHS = [
    "H:/flora_data/wav_resources/audioset_16k",
    "H:/flora_data/wav_resources/fma"
]
openwakeword.data.DEFAULT_RIR_PATHS = [
    "H:/flora_data/wav_resources/mit_rirs"
]

# This bypasses the ModuleNotFoundError by giving the library what it wants
if not hasattr(openwakeword, "resources"):
    class MockResources:
        RESOURCES = {
            "models": {
                "melspectrogram": os.path.join(OWW_RES_DIR, "melspectrogram.onnx"),
                "embedding": os.path.join(OWW_RES_DIR, "embedding_model.onnx")
            }
        }
    openwakeword.resources = MockResources
    print(f"✅ Manual Resource Injection: Models mapped to {OWW_RES_DIR}")


# 1. Import your working method (using the relative path that worked for you)
# Replace 'from .piper_sample_generator...' with your exact working line:
from .piper_sample_generator import generate_samples as working_method
mock_module = types.ModuleType("generate_samples")
mock_module.generate_samples = working_method
sys.modules["generate_samples"] = mock_module

print("✅ Hijack Complete: Method patched into mock module 'generate_samples'")

# FIX A: torchaudio.list_audio_backends

# FIX B: piper monotonic_align
try:
    import monotonic_alignment_search
    sys.modules["piper_train.vits.monotonic_align.monotonic_align"] = monotonic_alignment_search
except ImportError:
    print("Error: 'monotonic-alignment-search' missing. Run: poetry add monotonic-alignment-search")


# ==========================================
# 2. PATH & CONFIGURATION SETUP
# ==========================================

BASE_DIR = Path("H:/flora_data")
TARGET_WORD = "here kitty kitty"
MODEL_NAME = TARGET_WORD.replace(" ", "_")
CONFIG_PATH = "scratch/my_model.yaml"

def ensure_config():
    """Generates the YAML config with absolute H: drive paths if missing."""
    template_path = "scratch/custom_model.yml" # Assumes you copied this from site-packages
    
    if not os.path.exists(template_path):
        print(f"Error: {template_path} not found in project root.")
        return False

    with open(template_path, 'r') as f:
        config = yaml.load(f, yaml.Loader)

    # Inject our Flora-specific paths and settings
    config["target_phrase"] = [TARGET_WORD]
    config["model_name"] = MODEL_NAME
    config["n_samples"] = 1000 
    config["steps"] = 10000
    config["output_dir"] = str(Path("./my_custom_model").absolute())
    
    # H: Drive Data Paths
    config["background_paths"] = [str(BASE_DIR / "wav_resources/audioset_16k"), str(BASE_DIR / "wav_resources/fma")]
    config["rir_paths"] = [] #[str(BASE_DIR / "wav_resources/mit_rirs")]
    config["false_positive_validation_data_path"] = str(BASE_DIR / "features/validation_set_features.npy")
    config["feature_data_files"] = {
        "ACAV100M_sample": str(BASE_DIR / "features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy")
    }

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)
    print(f"✅ Configuration verified: {CONFIG_PATH}")
    return True


def harmonize_directories():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir")
    target_phrase = config.get("target_phrase")
    if isinstance(target_phrase, list):
        target_phrase = target_phrase[0]

    # Ensure the path handles spaces/special characters
    base_path = Path(output_dir) / target_phrase.replace(" ", "_")
    
    # Mapping: Old flat name -> New nested path
    mapping = {
        "positive_train": base_path / "train" / "positive",
        "negative_train": base_path / "train" / "negative",
        "positive_test":  base_path / "test"  / "positive",
        "negative_test":  base_path / "test"  / "negative"
    }

    print(f"🔄 Checking directory structure for '{target_phrase}'...")
    
    for old_name, new_path in mapping.items():
        old_path = base_path / old_name
        
        # If the old flat folder exists, migrate it
        if old_path.exists() and old_path.is_dir():
            print(f"  [!] Found legacy folder '{old_name}', migrating to standard structure...")
            new_path.mkdir(parents=True, exist_ok=True)
            
            # Move all .wav files from the old flat folder to the new nested one
            for wav_file in old_path.glob("*.wav"):
                shutil.move(str(wav_file), str(new_path / wav_file.name))
            
            # Remove the now-empty old folder
            old_path.rmdir()
            
            # CRITICAL: Create a 'ghost' file or empty folder so Phase 1 skip-logic stays happy
            # Phase 1 usually checks if the folder exists to decide whether to skip.
            old_path.mkdir(exist_ok=True) 

    print("✅ Directory structure is now harmonized for both Phases.")

# ==========================================
# 3. EXECUTION PHASES
# ==========================================

def run_oww_phase(phase_name, arg_flag):
    print(f"\n{'='*30}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*30}")
    
    # Set the arguments for the module
    sys.argv = [
        "openwakeword.train", 
        "--training_config", CONFIG_PATH, 
        arg_flag
    ]
    
    try:
        # run_module executes the code in the current process
        runpy.run_module("openwakeword.train", run_name="__main__")
        print(f"\n✅ {phase_name} completed successfully.")
    except Exception as e:
        print(f"\n❌ Error during {phase_name}: {e}")
        sys.exit(1)

def convert_to_tflite():
    print(f"\n{'='*30}")
    print(f"PHASE: TFLITE CONVERSION")
    print(f"{'='*30}")
    onnx_path = f"my_custom_model/{MODEL_NAME}.onnx"
    
    cmd = ["onnx2tf", "-i", onnx_path, "-o", "my_custom_model", "-kat", "onnx____Flatten_0"]
    subprocess.run(cmd, check=True)
    
    # Cleanup naming
    src = f"my_custom_model/{MODEL_NAME}_float32.tflite"
    dst = f"my_custom_model/{MODEL_NAME}.tflite"
    if os.path.exists(src):
        if os.path.exists(dst): os.remove(dst)
        os.rename(src, dst)
        print(f"✅ Model ready for Raspberry Pi: {dst}")

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def run():
    if ensure_config():
        #harmonize_directories()

        # Phase 1: Piper creates the voice samples
        #run_oww_phase("CLIP GENERATION", "--generate_clips")

        template_path = "scratch/my_model.yaml"
        with open(template_path, 'r') as f:
            config = yaml.load(f, yaml.Loader)

        print("\n🔍 DEBUG: Verifying Augmentation Paths...")
        for key in ['background_paths', 'rir_paths']:
            paths = config.get(key, [])
            print(paths)
            for p in paths:
                # Check for .wav files in the folder and all subfolders
                found_files = glob.glob(os.path.join(p, "**", "*.wav"), recursive=True)
                print(f"  - [{key}] Path: {p} | Files found: {len(found_files)}")
                if len(found_files) == 0:
                    print(f"  ⚠️ ALERT: No .wav files found in {p}!")
        
        # Phase 2: Mix with background noise/RIRs
        run_oww_phase("AUGMENTATION", "--augment_clips")
        
        # Phase 3: Train the actual model
        run_phase_train = run_oww_phase("TRAINING", "--train_model")
        
        # Phase 4: Export for the Pi
        convert_to_tflite()