from itertools import groupby
from typing import Any, cast, override

import numpy as np
import ai_parade.custom_runner as aip
import timm

def include():
	metadata_list = []
	models = sorted(timm.list_models(pretrained=True))
	unknown:list[list[str]] = []
	parsed:list[tuple[list[str], float | int, list[list[str]]]] = []

	splitted_models = [model.split("_") for model in models]
	for k, group in groupby(splitted_models, lambda x: x[0]):
		group_parsed, group_unknown = aip.parse_model_size(list(group), 1, 3)
		parsed += group_parsed
		unknown += group_unknown

	parsed += [(model, 1, [model]) for model in unknown]

	for family_name, base_size, models in parsed:
		for i, model_name in enumerate(models):
			model_name = "_".join(model_name)
			cache_dir = "models_cache"
			input_size = cast( list[int], timm.get_pretrained_cfg_value(model_name, "input_size"))
			assert input_size[0] == 3
			image_input = aip.ImageInput(
				batchSize=1, 
				height=input_size[1], 
				width=input_size[2], 
				channelOrder=aip.ImageInput.ChannelOrder.RGB, 
				dataOrder=aip.ImageInput.DataOrder.NCHW, 
				dataType=aip.ImageInput.DataType.FLOAT32, 
				)
			metadata = aip.ModelMetadataApi(
				name=model_name,
				family="_".join(family_name),
				format=aip.ModelFormat.PyTorch,
				task=aip.ModelTasks.Classification,
				image_input=image_input,
				size=base_size+(i/1000),
				get_runner=Runner,
				pytorch=aip.PyTorchOptionsApi(
					model_import="timm", 
					model_init_expression=f"module.create_model(model_name='{model_name}', pretrained=True, cache_dir='{cache_dir}', exportable=True)"
					),
				citation_link=cast( str, timm.get_pretrained_cfg_value(model_name, "origin_url")),
				citation=cast( str, timm.get_pretrained_cfg_value(model_name, "paper_name")),
				license=cast( str, timm.get_pretrained_cfg_value(model_name, "license")),
				)
			metadata_list.append(metadata)
	return metadata_list

def softmax(x: Any):
	return np.exp(x) / np.sum(np.exp(x))

class Runner(aip.PyTorchRunner):
	@override
	@staticmethod
	def _output_mapping(output: Any, model_metadata: aip.ModelMetadata) -> aip.ModelOutput:
		probabilities = softmax(output)
		max_index = np.argmax(probabilities)
		return {"objects":[{ "confidence": float(probabilities[max_index]), "ImageNet_1k_class": int(max_index) }]}
		return super()._output_mapping(output, model_metadata)