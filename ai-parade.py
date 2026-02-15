from itertools import groupby

import ai_parade.custom_definitions as aip
import timm

def include():
	sep_dataset = [x.split(".") for x in sorted(timm.list_models(pretrained=True))]
	model_2_dataset = {key: [(x[1] if len(x) == 2 else "") for x in group] for key, group in  groupby(sep_dataset, lambda x: x[0])}

	splitted_models = [x.split("_") for x in sorted(model_2_dataset.keys())]
	with_family, unknown_family = aip.parse_model_size(splitted_models, 1, 3)

	parsed = [
		("_".join(family), "_".join(model),  size + (i / len(models)))
		for family, size, models in with_family
		for i, model in enumerate(models)
	]

	parsed += [("_".join(models), "_".join(models), 1)
		for models in unknown_family
	]

	metadata_list = []
	for family, model_name, size in parsed:
			timm_ids = [(f"{model_name}.{dataset}" if dataset != "" else model_name) for dataset in model_2_dataset[model_name]]
			cfgs = [timm.get_pretrained_cfg(timm_id) for timm_id in timm_ids]
			cfgs = [cfg for cfg in cfgs if cfg is not None]
			download_links = {
				timm_id: f"https://huggingface.co/{cfg.hf_hub_id}/resolve/main/pytorch_model.bin" if cfg.hf_hub_id is not None else cfg.url
				for timm_id, cfg in zip(timm_ids, cfgs) if cfg.hf_hub_id is not None or cfg.url is not None
			}

			primary_idx = 0
			cfg = cfgs[primary_idx]
			assert cfg is not None
			assert cfg.input_size[0] == 3
			image_input = aip.ImageInput(
				batchSize=1, 
				height=cfg.input_size[1], 
				width=cfg.input_size[2], 
				channelOrder=aip.ImageInput.ChannelOrder.RGB, 
				dataOrder=aip.ImageInput.DataOrder.NCHW, 
				dataType=aip.ImageInput.DataType.float32,
				means=cfg.mean, # pyright: ignore[reportArgumentType]
				stds=cfg.std, # pyright: ignore[reportArgumentType]
				)
			metadata = aip.ModelMetadataApi(
				name=model_name,
				family=family,
				size=size,
				format=aip.ExactModelFormat(aip.ModelFormat.PyTorch),
				task=aip.ModelTasks.Classification,
				citation=cfg.paper_name,
				citation_link=cfg.origin_url,
				download_link=download_links, # pyright: ignore[reportArgumentType]
				license=cfg.license,
				image_input=image_input,
				output="",
				dataset=model_2_dataset[model_name],
				pytorch=aip.PyTorchOptionsApi(
					model_import="timm", 
					model_init_expression=f"module.create_model(model_name='{timm_ids[primary_idx]}', pretrained=False, exportable=True)"
					),
				)
			metadata_list.append(metadata)
	return metadata_list