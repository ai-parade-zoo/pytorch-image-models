from typing import Any, cast, override

import numpy as np
import ai_parade.custom_runner as aip
import timm

def include(config: aip.Config):
	metadata_list = []
	for model_name in timm.list_models(pretrained=True):
		cache_dir = config.models_directory/"pytorch-image-models"
		input_size = cast( list[int], timm.get_pretrained_cfg_value(model_name, "input_size"))
		image_input = aip.ImageInput(batchSize=1, height=input_size[0], width=input_size[1], channelOrder=aip.ImageInput.ChannelOrder.RGB, dataOrder=aip.ImageInput.DataOrder.NCHW, dataType=aip.ImageInput.DataType.FLOAT32, )
		metadata = aip.ModelMetadataApi(
				name=model_name,
				format=aip.ModelFormat.PyTorch,
				task=aip.ModelTasks.Classification,
				image_input=image_input,
				runner_type=Runner,
				pytorch=aip.PyTorchOptionsApi(
					model_import="timm", 
					model_init_expression=f"module.create_model(model_name='{model_name}', pretrained=True, cache_dir='{cache_dir}', exportable=True)"
					)
				)
		metadata_list.append(metadata)
	return metadata_list

def softmax(x: Any):
	return np.exp(x) / np.sum(np.exp(x))

class Runner(aip.PyTorchRunner):
	@override
	@classmethod
	def _output_mapping(cls, output: Any, model_metadata: aip.ModelMetadata) -> aip.ModelOutput:
		probabilities = softmax(output)
		max_index = np.argmax(probabilities)
		return {"objects":[{ "confidence": float(probabilities[max_index]), "ImageNet_1k_class": int(max_index) }]}
		return super()._output_mapping(output, model_metadata)