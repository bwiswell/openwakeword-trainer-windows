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


## Note

This README is obviously very incomplete. More to follow - just note that there is ~50GB of data that this pipeline will download/extract - you'll need a lot of space.


## Acknowledgements

Please reference the [openWakeWord](https://github.com/dscripka/openWakeWord) repository - openWakeWord was created by [dscripka](https://github.com/dscripka) and made available under the **Apache 2.0** license. I did not create the model - just this training pipeline, which itself relies heavily on the original repository.