import os

from src.data.dataset_hf_path import EVAL_DATASET_HF_PATH
from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook, RESOLUTION_MAPPING, ImageVideoInstance
from src.data.utils.dataset_utils import load_hf_dataset, sample_dataset
from src.data.utils.vision_utils import save_frames, process_video_frames, VID_EXTENSIONS
from src.model.processor import process_input_text


TASK_INST_QRY = "Find the clip that corresponds to the described scene in the given video:"
TASK_INST_TGT = "Understand the content of the provided video."

@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    image_resolution = kwargs['image_resolution']
    max_video_frames_saved = kwargs["max_video_frames_saved"]
    max_clip_frames_saved = kwargs["max_clip_frames_saved"]
    num_video_frames = kwargs["num_video_frames"]
    num_clip_frames = kwargs["num_clip_frames"]
    model_backbone = kwargs["model_backbone"]
    video_root, clip_root, frame_root = kwargs["video_root"], kwargs["clip_root"], kwargs["frame_root"]

    query_texts, query_images, cand_texts, cand_images, dataset_infos = [], [], [], [], []
    
    for query, query_video_path, clips_dir_path in \
        zip(batch_dict['query'], batch_dict['video_path'], batch_dict['clips_dir_path']):    

        video_name = os.path.splitext(os.path.basename(query_video_path))[0]
        frames_dir = os.path.join(frame_root, video_name)

        # Load query video
        query_video_path = os.path.join(video_root, os.path.basename(query_video_path)) if video_root else None
        query_frame_dir = os.path.join(frames_dir, "query")
        save_frames(video_path=query_video_path,
                    frame_dir=query_frame_dir,
                    max_frames_saved=max_video_frames_saved)
        qry_frame_paths = process_video_frames(query_frame_dir, num_frames=num_video_frames)

        query_texts.append([process_input_text(TASK_INST_QRY, model_backbone, text=query, add_video_token=True)])
        query_images.append([ImageVideoInstance(
            bytes=[None] * len(qry_frame_paths),
            paths=qry_frame_paths,
            resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(qry_frame_paths),
        ).to_dict()])

        # Load pos and neg clip, save the frames if only the raw video is provided.
        cand_names, cand_frames = [], []
        clip_video_dir = os.path.join(clip_root, video_name) if clip_root else None
        clip_video_paths = [f for f in os.listdir(clip_video_dir) if os.path.splitext(f)[1].lower() in VID_EXTENSIONS]
        for clip_video_path in clip_video_paths:
            clip_name = os.path.splitext(clip_video_path)[0]
            clip_frame_dir_or_file = os.path.join(frames_dir, clip_name)
            clip_video_path_abs = os.path.join(clip_video_dir, clip_video_path)
            save_frames(video_path=clip_video_path_abs,
                        frame_dir=clip_frame_dir_or_file,
                        max_frames_saved=max_clip_frames_saved)
        for clip_frame_dir_or_file in os.listdir(frames_dir):
            clip_frame_dir_abs = os.path.join(frames_dir, clip_frame_dir_or_file)
            if clip_frame_dir_or_file == 'query' or os.path.isfile(clip_frame_dir_abs):
                continue
            cand_names.append(clip_frame_dir_abs)  # use absolute path here instead of file name to keep it unique
            if clip_frame_dir_or_file.startswith("positive"):
                label_name = clip_frame_dir_abs
            cand_frame_paths = process_video_frames(clip_frame_dir_abs, num_frames=num_clip_frames)
            cand_images.append([ImageVideoInstance(
                bytes=[None] * len(cand_frame_paths),
                paths=cand_frame_paths,
                resolutions=[RESOLUTION_MAPPING.get(image_resolution, None)] * len(cand_frame_paths),
            ).to_dict()])

        cand_texts.append([process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)] * len(cand_names))
        cand_images.append(cand_frames)
        dataset_infos.append({
            "cand_names": cand_names,
            "label_name": label_name,
        })

    return {"query_text": query_texts, "query_image": query_images,
            "cand_text": cand_texts, "cand_image": cand_images,
            "dataset_infos": dataset_infos}


DATASET_PARSER_NAME = "moment_retrieval"
@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_moment_retrieval_dataset(model_args, data_args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['model_backbone'] = model_args.model_backbone
    kwargs['image_resolution'] = data_args.image_resolution
    
    dataset = dataset.map(lambda x: data_prepare(x, **kwargs), batched=True, batch_size=64,
                          drop_last_batch = False)
    dataset = dataset.select_columns(["query_text", "query_image", "cand_text", "cand_image", "dataset_infos"])
    corpus = None  # No additional corpus

    return dataset, corpus
