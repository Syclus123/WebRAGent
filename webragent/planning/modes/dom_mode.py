import copy
import logging

from agent.Prompt import *
from agent.LLM import *
from .base_mode import InteractionMode

logger = logging.getLogger(__name__)


class DomMode(InteractionMode):
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)
    
    async def execute(self, status_description, user_request, rag_enabled, rag_path, previous_trace, observation, feedback, observation_VforD, rag_mode="description", rag_cache_dir=None):
        rag_data = {
            "rag_enabled": rag_enabled,
            "rag_path": rag_path if rag_enabled else None,
            "rag_mode": rag_mode
        }

        if rag_enabled:
            # Select RAG constructor based on mode
            if rag_mode == "vision_rag":
                from agent.Prompt.prompt_constructor import DOMVisionRAGConstructor
                prompt_constructor = DOMVisionRAGConstructor()
                
                # DOMVisionRAGConstructor needs screenshot_base64 parameter (like OperatorVisionRAGConstructor)
                planning_request = prompt_constructor.construct(
                    user_request=user_request,
                    rag_path=rag_path,
                    previous_trace=previous_trace,
                    observation=observation,
                    feedback=feedback,
                    status_description=status_description,
                    screenshot_base64=observation_VforD,  # Pass observation_VforD as screenshot_base64
                    rag_config={} if not rag_cache_dir else {"rag_cache_dir": rag_cache_dir},
                    rag_cache_dir=rag_cache_dir or ""
                )
                
                # Record retrieved info for logging
                if hasattr(prompt_constructor, 'get_last_retrieved_info'):
                    retrieved_info = prompt_constructor.get_last_retrieved_info()
                    if retrieved_info:
                        rag_data["retrieved_info"] = {
                            "cand_id": retrieved_info.get('cand_id', ''),
                            "task_description": retrieved_info.get('task_description', ''),
                            "similarity_score": retrieved_info.get('score', 0.0)
                        }
                        
            elif rag_mode == "vision":
                # Use existing PlanningPromptVisionRetrievalConstructor
                prompt_constructor = PlanningPromptVisionRetrievalConstructor()
                planning_request = prompt_constructor.construct(
                    user_request, rag_path, previous_trace, observation, feedback, status_description)
            else:
                # Default to description RAG
                prompt_constructor = PlanningPromptDescriptionRetrievalConstructor()
                planning_request = prompt_constructor.construct(
                    user_request, rag_path, previous_trace, observation, feedback, status_description)

            rag_data["rag_method"] = prompt_constructor.__class__.__name__
            
            # Record the retrieved example information (from prompt_constructor)
            # Note: DOMVisionRAGConstructor uses different retrieval info format
            if rag_mode != "vision_rag" and hasattr(prompt_constructor, 'reference'):
                reference = getattr(prompt_constructor, 'reference', None)
                if reference:
                    rag_data["retrieved_examples"] = reference
        else:
            planning_request = PlanningPromptConstructor().construct(user_request, previous_trace, observation, feedback, status_description)

        planning_request_copy = copy.deepcopy(planning_request)
        rag_data["planning_request"] = planning_request_copy

        # Filter out base64 image data for log size optimization
        def filter_base64_from_messages(messages):
            """Remove base64 image data from messages for logging"""
            filtered_messages = []
            for msg in messages:
                filtered_msg = {"role": msg["role"]}
                if msg["role"] == "system":
                    filtered_msg["content"] = msg["content"]
                elif msg["role"] == "user" and isinstance(msg["content"], list):
                    filtered_content = []
                    for item in msg["content"]:
                        if item["type"] == "text":
                            # Keep full text content as requested by user
                            filtered_content.append({
                                "type": "text",
                                "text": item["text"]
                            })
                        elif item["type"] == "image_url":
                            filtered_content.append({
                                "type": "image_url",
                                "image_url": {"url": "[base64_image_data_removed_for_log_size]"}
                            })
                        else:
                            filtered_content.append(item)
                    filtered_msg["content"] = filtered_content
                else:
                    filtered_msg["content"] = msg["content"]
                filtered_messages.append(filtered_msg)
            return filtered_messages
        
        filtered_planning_request = filter_base64_from_messages(planning_request)
        logger.info(
            f"\033[32mDOM_based_planning_request:\n{filtered_planning_request}\033[0m\n")
        logger.info(f"planning_text_model: {self.text_model.model}")
        planning_response, error_message = await self.text_model.request(planning_request)

        # Logging the response
        rag_data["planning_response"] = planning_response

        input_token_count = calculation_of_token(planning_request, model=self.text_model.model)
        output_token_count = calculation_of_token(planning_response, model=self.text_model.model)
        planning_token_count = [input_token_count, output_token_count]

        rag_data["token_counts"] = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count
        }

        return planning_response, error_message, None, None, planning_token_count, rag_data
