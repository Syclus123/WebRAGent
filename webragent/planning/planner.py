import time
import logging

from agent.LLM import *
from agent.Prompt import *
from ..utils.rag_logger import RAGLogger, VisionRAGLogger
from .action_parser import ActionParser, ResponseError
from .modes import (
    InteractionMode, DomMode, DomVDescMode,
    VisionToDomMode, VisionMode, DVMode, OperatorMode,
)

logger = logging.getLogger(__name__)


class Planner:

    @staticmethod
    async def plan(
        config,
        user_request,
        text_model_name,
        previous_trace,
        observation,
        feedback,
        mode,
        observation_VforD,
        status_description,
        rag_enabled,
        rag_path,
        rag_mode="description",
        rag_log_dir=None,
        rag_cache_dir=None
    ):

        # select rag logger
        rag_logger = None
        logger_method = None
        if rag_log_dir is not None:
            if rag_enabled and rag_mode == "vision":
                rag_logger = VisionRAGLogger(rag_log_dir=rag_log_dir)
                logger_method = "log_vision_rag_step"
            else:
                rag_logger = RAGLogger(rag_log_dir=rag_log_dir)
                logger_method = "log_rag_step"
        
        # Get the current step index from previous_trace
        step_idx = len(previous_trace)
        # task id
        task_id = f"{user_request[:50]}_{int(time.time())}"
        if hasattr(config, 'task_id') and config.task_id:
            task_id = config.task_id
        
        fallback_text_model = config.get("model", {}).get("fallback_text_model", "gpt-3.5-turbo")
        fallback_visual_model = config.get("model", {}).get("fallback_visual_model", "gpt-4-turbo")
        gpt35 = GPTGenerator(model=fallback_text_model)
        gpt4v = GPTGenerator(model=fallback_visual_model)

        all_json_models = config["model"]["json_models"]
        is_json_response = config["model"]["json_model_response"]

        llm_planning_text = create_llm_instance(
            text_model_name, is_json_response, all_json_models)

        modes = {
            "dom": DomMode(text_model=llm_planning_text),
            "dom_v_desc": DomVDescMode(visual_model=gpt4v, text_model=llm_planning_text),
            "vision_to_dom": VisionToDomMode(visual_model=gpt4v, text_model=llm_planning_text),
            "d_v": DVMode(visual_model=gpt4v),
            "vision": VisionMode(visual_model=gpt4v),
            "operator": OperatorMode(text_model=llm_planning_text)  # Add operator mode
        }

        if mode == "operator":
            # prompt logging parameters check
            prompt_logging_enabled = getattr(config, 'prompt_logging_enabled', False)
            prompt_logger = getattr(config, 'prompt_logger', None)
            task_uuid = getattr(config, 'task_uuid', None)
            step_idx = getattr(config, 'step_idx', 0)
            
            result = await modes[mode].execute(
                status_description=status_description,
                user_request=user_request,
                rag_enabled=rag_enabled,
                rag_path=rag_path,
                previous_trace=previous_trace,
                observation=observation,
                feedback=feedback,
                observation_VforD=observation_VforD,
                rag_mode=rag_mode,
                prompt_logging_enabled=prompt_logging_enabled,
                prompt_logger=prompt_logger,
                task_uuid=task_uuid,
                step_idx=step_idx)
        else:
            if mode == "dom":
                # DOM mode supports RAG mode parameter
                result = await modes[mode].execute(
                    status_description=status_description,
                    user_request=user_request,
                    rag_enabled=rag_enabled,
                    rag_path=rag_path,
                    previous_trace=previous_trace,
                    observation=observation,
                    feedback=feedback,
                    observation_VforD=observation_VforD,
                    rag_mode=rag_mode,
                    rag_cache_dir=rag_cache_dir)
            else:
                # Other modes use original signature
                result = await modes[mode].execute(
                    status_description=status_description,
                    user_request=user_request,
                    rag_enabled=rag_enabled,
                    rag_path=rag_path,
                    previous_trace=previous_trace,
                    observation=observation,
                    feedback=feedback,
                    observation_VforD=observation_VforD)
        
        # Check if any RAG data is returned
        if len(result) >= 6 and mode in ["dom", "operator"]:  # Both DomMode and OperatorMode return rag_data
            planning_response, error_message, planning_response_thought, planning_response_action, planning_token_count, rag_data = result
        
            rag_data["mode"] = mode
            rag_data["user_request"] = user_request
            
            if rag_logger is not None and logger_method is not None:
                if logger_method == "log_vision_rag_step":
                    rag_logger.log_vision_rag_step(task_id, step_idx, rag_data)
                else:
                    rag_logger.log_rag_step(task_id, step_idx, rag_data)
        else:
            # Compatible with other patterns that do not return rag_data
            planning_response, error_message, planning_response_thought, planning_response_action, planning_token_count = result
        
            # log
            rag_data = {
                "mode": mode,
                "user_request": user_request,
                "rag_enabled": False
            }
            
            if rag_logger is not None and logger_method is not None:
                if logger_method == "log_vision_rag_step":
                    rag_logger.log_vision_rag_step(task_id, step_idx, rag_data)
                else:
                    rag_logger.log_rag_step(task_id, step_idx, rag_data)

        logger.info(f"\033[34mPlanning_Response:\n{planning_response}\033[0m")
        
        # Handle operator mode differently - it already parses the response
        if mode != "vision_to_dom" and mode != "operator":
            try:
                planning_response_thought, planning_response_action = ActionParser().extract_thought_and_action(
                    planning_response)
            except ResponseError as e:
                logger.error(f"Response Error:{e.message}")
                raise

        # Special handling for fill_form -> fill_search conversion
        if planning_response_action.get('action') == "fill_form":
            JudgeSearchbarRequest = JudgeSearchbarPromptConstructor().construct(
                input_element=observation, planning_response_action=planning_response_action)
            try:
                Judge_response, error_message = await gpt35.request(JudgeSearchbarRequest)
                if Judge_response.lower() == "yes":
                    planning_response_action['action'] = "fill_search"
            except Exception:
                planning_response_action['action'] = "fill_form"

        # The description should include both the thought (returned by LLM) and the action (parsed from the planning response)
        planning_response_action["description"] = {
            "thought": planning_response_thought,
            "action": (
                f'{planning_response_action["action"]}: {planning_response_action["action_input"]}' if "description" not in planning_response_action.keys() else
                planning_response_action["description"])
            if mode in ["dom","d_v", "dom_v_desc", "vision_to_dom", "operator"] else (
                planning_response_action["action"] if "description" not in planning_response_action.keys() else
                planning_response_action["description"])
        }
        
        # Format action based on mode
        if mode in ["dom", "d_v", "dom_v_desc", "vision_to_dom", "operator"]:
            planning_response_action = {element: planning_response_action.get(
                element, "") for element in ["element_id", "action", "action_input", "description"]}
        elif mode == "vision":
            planning_response_action = {element: planning_response_action.get(
                element, "") for element in ["action", "description"]}
        
        logger.info("****************")
        # logger.info(planning_response_action)
        dict_to_write = {}
        if mode in ["dom", "d_v", "dom_v_desc", "vision_to_dom", "operator"]:
            dict_to_write['id'] = planning_response_action['element_id']
            dict_to_write['action_type'] = planning_response_action['action']
            dict_to_write['value'] = planning_response_action['action_input']
        elif mode == "vision":
            dict_to_write['action'] = planning_response_action['action']
        dict_to_write['description'] = planning_response_action['description']
        dict_to_write['error_message'] = error_message
        dict_to_write['planning_token_count'] = planning_token_count

        return dict_to_write


# Backward compatibility alias
Planning = Planner
