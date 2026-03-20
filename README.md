# openwakeword-trainer-windows

[OpenWakeWord](https://openwakeword.com/) training pipeline that works on Windows - because NT should mean that you get to have Nice Things (please learn how to use WSL - there are way too many monkey patches required for this to work, and it is unlikely to be a stable solution).


## Dependencies

### Required
- poetry

### Optional
- cuDNN 12.9 if you want to use GPU (highly recommended if you have a GPU device, MUST be cuDNN 12.9)


## Setup

- Clone the repo
- `cd` into the repo
- Install deps with poetry (`poetry install`)
- Duplicate `configs/example.yaml` and rename it as desired (use underscores instead of spaces)
- Edit the contents of your new `configs/your_model_name.yaml`
- Run with `poetry run python -m openwakeword_trainer_windows your_model_name`
- ...profit? Or at least use your custom home assistant


## CLI Arguments

`poetry run python -m openwakeword_trainer_windows <model-name> -d <data-path> -o <output-path> -s <start-from> -e <end-at> -i <only-do>`

- `<model-name>`: The model to train (should be the same as your `.yaml` name, use underscores instead of spaces).
- `<data-path>`: Optional, path where downloaded datasets and features should be stored (~50GB), as well as recorded/generated TTS samples. Defaults to `<repo-directory>/data`.
- `<output-path>`: Optional, path where the final `.onnx` and `.tflite` models should be stored. Defaults to `<repo-directory>/output`.
- `<start-from>`: Optional, pipeline phase to start execution from. See [Pipeline Phases](#pipeline-phases) for available options. Defaults to `'ensure'`.
- `<end-at>`: Optional, pipeline phase to end execution at (inclusive). See [Pipeline Phases](#pipeline-phases) for available options. Defaults to `'export'`.
- `<only-do>`: Optional, if passed then only this phase will be executed (even if `<start-from>` or `<end-at>` is passed). See [Pipeline Phases](#pipeline-phases) for available options. No default.


## Pipeline Phases

This is a high-level overview of the steps that the training pipeline takes, with a brief description of what each step does and the corresponding value that can be passed as `<start-from>`, `<end-at>`, or `<only-do>` (see [CLI Arguments](#cli-arguments)).

### Ensure paths (`'ensure'`):

Creates the directories used for training data and pipeline outputs. **This will remove existing TTS training data from previous runs for the provided model** (but not downloaded datasets or recorded samples).

### Download resources (`'download'`):

Downloads `openwakeword`, remote training features, models, and datasets. Each download is automatically skipped if it already exists in the expected location.

### Unpack resources (`'unpack'`):

Installs `openwakeword`, copies the models that `openwakeword` depends on into the appropriate directory, and extracts `.wav` training data from the downloaded datasets. Each extraction is automatically skipped if it was already performed on a prior run.

### Patch `openwakeword` (`'patch'`):

Applies numerous compatibility patches that allow `openwakeword` to work seamlessly on Windows in the built `poetry` environment. Each patch is automatically skipped if it was already applied.

### Create configuration file (`'configure'`):

Reads the data from the user-created `openwakeword-trainer-windows`-compatible `.yaml` file and builds an `openwakeword`-compatible `.yaml` configuration file to use for data augmentation and training operations.

### Record samples (`'record'`):

Provides a CLI interface for the user to record samples directly from a device microphone. Recorded samples for each configured model are saved and reused between training runs.

### Generate TTS samples (`'tts'`):

Generates artifical TTS samples for training the model. **These samples will be deleted each time the [ensure paths](#ensure-paths-ensure) step is run**.

### Augment data (`'augment'`):

Augments the user-recorded and TTS-generated training samples by mixing them with free-to-use music and room impulse samples.

### Train model (`'train'`):

Trains the `openwakeword` model using the user-provided `.yaml` configuration and the recorded/generated training samples.

### Export models/stats (`'export'`):

Converts the resulting `.onnx` model to `.tflite` and moves both `.onnx` and `.tflite` results to the output directory (along with a `.json` file containing accuracy, false positive rate, and recall metrics).


## Note

This README is obviously very incomplete. More to follow - just note that there is ~50GB of data that this pipeline will download/extract - you'll need a lot of space.


## Acknowledgements

Please reference the [openWakeWord](https://github.com/dscripka/openWakeWord) repository - openWakeWord was created by [dscripka](https://github.com/dscripka) and made available under the **Apache 2.0** license. I did not create the model - just this training pipeline, which itself relies heavily on the original repository.
