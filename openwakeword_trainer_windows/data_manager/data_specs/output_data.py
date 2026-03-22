from dataclasses import dataclass
from pathlib import Path

from .data_spec import DataSpec


@dataclass
class OutputData(DataSpec):

    exports: str
    onnx_export: str
    onnx_output: str
    stats: str
    tflite_export: str
    tflite_output: str

    def __init__ (self, export_path: Path, output_path: Path, model_name: str):
        DataSpec.__init__(
            self,
            final_path = output_path,
            recreates = [export_path, output_path]
        )
        self.exports = str(export_path)
        self.onnx_export = str(export_path / f'{model_name}.onnx')
        self.onnx_output = str(output_path / f'{model_name}.onnx')
        self.stats = str(output_path / f'{model_name}.json')
        self.tflite_export = str(export_path / f'{model_name}_float32.tflite')
        self.tflite_output = str(output_path / f'{model_name}.tflite')