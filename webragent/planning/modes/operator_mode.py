import time
import json
import json5
import os
import re
import traceback
import logging
from typing import List, Dict, Any

from agent.Prompt import *
from agent.LLM import *
from agent.Prompt.prompt_constructor import OperatorPromptConstructor, OperatorPromptRAGConstructor
from agent.Prompt.operator_prompts import OperatorPrompts
from agent.Utils.utils import is_valid_base64
from .base_mode import InteractionMode

logger = logging.getLogger(__name__)


class OperatorMode(InteractionMode):
    """
    OpenAI Operator mode for browser automation
    """
    
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)
        # Operator is the primary model for this mode
        self.operator_model = text_model
        self.conversation_history = []
        self.current_screenshot = None
        
        # Cache RAG constructors to avoid re-initialization
        self.rag_constructors = {}
    
    async def execute(self, status_description, user_request, rag_enabled, rag_path, 
                     previous_trace, observation, feedback, observation_VforD, rag_mode="description",
                     prompt_logging_enabled=False, prompt_logger=None, task_uuid=None, step_idx=0, rag_cache_dir=None):
        """
        Execute operator planning with proper OpenAI Operator integration
        
        Args:
            status_description: Current task status
            user_request: User's task request
            rag_enabled: Whether RAG is enabled
            rag_path: Path to RAG data
            previous_trace: Previous action history
            observation: Current DOM observation (not used in operator mode)
            feedback: Any feedback or error messages
            observation_VforD: Screenshot in base64 format
            rag_mode: RAG mode - "description" or "vision" (default: "description")
            prompt_logging_enabled: Whether to enable prompt logging
            prompt_logger: PromptLogger instance
            task_uuid: Task UUID for logging
            step_idx: Current step index for logging
            rag_cache_dir: RAG cache directory for pre-built indices
            
        Returns:
            Tuple of (planning_response, error_message, planning_response_thought, 
                     planning_response_action, planning_token_count, rag_data)
        """
        # record execution start time for prompt logging
        start_time = time.time()
        
        rag_data = {
            "rag_enabled": rag_enabled,
            "rag_path": rag_path if rag_enabled else None,
            "rag_mode": rag_mode,
            "mode": "operator"
        }
        
        # init RAG data
        self.current_rag_data = rag_data.copy()
        
        try:
            # Store current screenshot for conversation continuity
            self.current_screenshot = observation_VforD
            
            # Build conversation messages
            messages = self._build_operator_messages(user_request, status_description, 
                                                   previous_trace, feedback, rag_enabled, rag_path, rag_mode, rag_cache_dir)
            
            # update rag_data
            rag_data.update(self.current_rag_data)
            
            # Log the planning request
            logger.info(f"\033[36mOperator Planning Request:\n{user_request}\033[0m")
            logger.info(f"Operator Model: {self.operator_model.model}")
            logger.info(f"Screenshot Available: {observation_VforD is not None}")
            if rag_enabled:
                logger.info(f"🧠 RAG Mode: {rag_data.get('rag_constructor_type', 'Unknown')}")
                # logger.info(f"📚 RAG Reference Available: {bool(rag_data.get('rag_reference', ''))}")
            
            # Make request to OpenAI Operator
            planning_response, error_message = await self.operator_model.request(
                messages=messages,
                screenshot_base64=observation_VforD,
                viewport_width=1280,
                viewport_height=720
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            if error_message:
                logger.error(f"Operator API Error: {error_message}")
                rag_data["error"] = error_message
                
                # prompt logging(failed case)
                if prompt_logging_enabled and prompt_logger and task_uuid is not None:
                    try:
                        prompt_data = {
                            "model": getattr(self.operator_model, 'model', 'unknown'),
                            "rag_mode": rag_mode,
                            "rag_enabled": rag_enabled,
                            "user_request": user_request,
                            "status_description": status_description,
                            "feedback": feedback,
                            "previous_trace": previous_trace,
                            "screenshot_available": observation_VforD is not None,
                            "messages": messages,
                            "input_tokens": len(str(messages)) // 4,
                            "output_tokens": 0,
                            "planning_response": "",
                            "planning_thought": "",
                            "planning_action": {"action": "operator_wait", "action_input": "1000"},
                            "execution_time_ms": execution_time_ms,
                            "error": error_message
                        }
                        # add RAG related data
                        prompt_data.update(rag_data)
                        prompt_logger.log_prompt_step(task_uuid, step_idx, prompt_data)
                        logger.info(f"📝 Prompt logged for error case: step {step_idx}")
                    except Exception as log_error:
                        logger.warning(f"⚠️  Failed to log error prompt: {log_error}")
                
                return ("", error_message, "", {"action": "operator_wait", "action_input": "1000", "ms": 1000, "element_id": "api_error_wait"}, [0, 0], rag_data)
            
            # Log the response
            logger.info(f"\033[32mOperator Response:\n{planning_response}\033[0m")
            rag_data["planning_response"] = planning_response
            
            # Parse the operator response
            try:
                # Validate planning_response before parsing
                if not planning_response:

                    logger.error("⚠️ Empty or None planning response received")
                    raise ValueError("Empty or None planning response")
                
                if not isinstance(planning_response, str):
                    logger.error(f"⚠️ Invalid planning response type: {type(planning_response)}")
                    raise ValueError(f"Invalid planning response type: {type(planning_response)}")
                
                # Clean the response string
                planning_response = planning_response.strip()
                if not planning_response:
                    logger.error("⚠️ Empty planning response after stripping")
                    raise ValueError("Empty planning response after stripping")
                
                # Try to parse JSON
                try:
                    response_data = json.loads(planning_response)
                except json.JSONDecodeError as json_error:
                    logger.error(f"⚠️ JSON decode error: {json_error}")
                    logger.error(f"Raw response: {repr(planning_response)}")
                    # Try to extract action from text if JSON parsing fails
                    response_data = {"text_response": planning_response, "actions": [], "reasoning": "JSON parsing failed"}
                
                # Validate response_data
                if response_data is None:
                    logger.error("⚠️ Parsed response_data is None")
                    response_data = {"text_response": planning_response, "actions": [], "reasoning": "Parsed data is None"}
                
                # ensure all fields are converted to empty strings even if they are null
                actions = response_data.get("actions", []) or []
                text_response = response_data.get("text_response", "") or ""
                reasoning = response_data.get("reasoning", "") or ""
                
                # Log raw responses for debugging purposes
                logger.debug(f"🔍 Operator API Response Fields: {list(response_data.keys()) if response_data else 'None'}")
                logger.debug(f"📝 text_response: '{(text_response or '')[:100]}{'...' if len(text_response or '') > 100 else ''}'")
                logger.debug(f"🧠 reasoning: '{(reasoning or '')[:100]}{'...' if len(reasoning or '') > 100 else ''}'")
                logger.debug(f"🎯 actions count: {len(actions) if actions else 0}")
                
                # Improve thinking content extraction logic - try more fields
                # Ensure all potential thoughts are strings (handle None values)
                potential_thoughts = [
                    text_response or "",
                    reasoning or "",
                    response_data.get("explanation", "") or "",
                    response_data.get("thought", "") or "", 
                    response_data.get("description", "") or "",
                    response_data.get("plan", "") or ""
                ]
                
                # Filter out any remaining None values as extra safety
                potential_thoughts = [str(t) if t is not None else "" for t in potential_thoughts]
                
                # Find the first non-empty thought
                planning_response_thought = ""
                for thought in potential_thoughts:
                    if thought and isinstance(thought, str) and thought.strip():
                        planning_response_thought = thought.strip()
                        break
                
                # If there is still no thought content, generate a more meaningful description based on the action
                if not planning_response_thought:
                    if actions:
                        first_action = actions[0]
                        action_data = first_action.get("action", {})
                        action_type = action_data.get("type", "wait")
                        
                        # Generating descriptive thoughts based on action types
                        action_descriptions = {
                            "click": "Clicking on an element to interact with it",
                            "type": "Typing text into an input field",
                            "scroll": "Scrolling to view more content",
                            "wait": "Waiting for page elements to load",
                            "screenshot": "Taking a screenshot to analyze current state",
                            "key": "Pressing keyboard keys for navigation"
                        }
                        
                        planning_response_thought = action_descriptions.get(
                            action_type, 
                            f"Executing {action_type} action"
                        )
                        
                        logger.info(f"💭 Generated thought from action: '{planning_response_thought}'")
                    else:
                        # The final alternative
                        planning_response_thought = f"Processing step with status: {status_description[:50]}{'...' if len(status_description) > 50 else ''}"
                        logger.warning("⚠️ No thought content available, using status-based description")
                else:
                    logger.info(f"✅ Found thought content: '{planning_response_thought[:50]}{'...' if len(planning_response_thought) > 50 else ''}'")
                
                # Convert operator actions to compatible format
                if actions and isinstance(actions, list) and len(actions) > 0:
                    # Take the first action for now
                    first_action = actions[0]
                    
                    # response format: {"action": {"type": "click", "x": 773, "y": 90}}
                    action_data = first_action.get("action", {})
                    planning_response_action = self._convert_operator_action(action_data)
                    
                    logger.info(f"🔧 Converted action: {planning_response_action}")
                else:
                    # No actions returned, try to parse action from text_response
                    logger.warning("⚠️ No valid actions found in response, parsing from text")
                    planning_response_action = self._parse_action_from_text(text_response)
                
                # Calculate token counts (correctly)
                input_token_count = calculation_of_token(messages, model=self.operator_model.model)
                output_token_count = calculation_of_token(planning_response, model=self.operator_model.model)
                planning_token_count = [input_token_count, output_token_count]
                
                # Store in conversation history for continuity
                self.conversation_history.append({
                    "messages": messages,
                    "response": planning_response,
                    "actions": actions
                })
                
                # prompt logging(success case)
                if prompt_logging_enabled and prompt_logger and task_uuid is not None:
                    try:
                        prompt_data = {
                            "model": getattr(self.operator_model, 'model', 'unknown'),
                            "rag_mode": rag_mode,
                            "rag_enabled": rag_enabled,
                            "user_request": user_request,
                            "status_description": status_description,
                            "feedback": feedback,
                            "previous_trace": previous_trace,
                            "screenshot_available": observation_VforD is not None,
                            "messages": messages,
                            "input_tokens": planning_token_count[0] if planning_token_count else 0,
                            "output_tokens": planning_token_count[1] if planning_token_count else 0,
                            "planning_response": planning_response,
                            "planning_thought": planning_response_thought,
                            "planning_action": planning_response_action,
                            "execution_time_ms": execution_time_ms
                        }
                        # add RAG related data
                        prompt_data.update(rag_data)
                        prompt_logger.log_prompt_step(task_uuid, step_idx, prompt_data)
                        logger.info(f"📝 Prompt logged successfully: step {step_idx}")
                    except Exception as log_error:
                        logger.warning(f"⚠️  Failed to log prompt: {log_error}")
                
                return (planning_response, error_message, planning_response_thought, 
                       planning_response_action, planning_token_count, rag_data)
                
            except Exception as e:
                logger.error(f"❌ Error parsing operator response: {e}")
                logger.error(f"📝 Raw planning response: {repr(planning_response)}")
                logger.error(f"🧠 RAG mode: {rag_mode}")
                logger.error(f"📷 Screenshot available: {observation_VforD is not None}")
                
                # Enhanced error handling with more context
                error_context = f"Parsing error in {rag_mode} mode"
                if rag_mode == "vision_rag":
                    error_context += " - Check RAG database initialization and embedding model"
                elif rag_mode == "vision":
                    error_context += " - Check vision RAG retrieval process"
                
                planning_response_thought = f"Error parsing operator response: {error_context}"
                planning_response_action = {"action": "operator_wait", "action_input": "1000", "ms": 1000, "element_id": "parse_error_wait"}
                
                # Add error details to rag_data for debugging
                rag_data["parse_error"] = {
                    "error": str(e),
                    "raw_response": str(planning_response)[:500] if planning_response else "None",
                    "response_type": type(planning_response).__name__,
                    "rag_mode": rag_mode,
                    "screenshot_available": observation_VforD is not None
                }
                
                return (planning_response or "", str(e), planning_response_thought, 
                       planning_response_action, [0, 0], rag_data)
            
        except Exception as e:
            logger.error(f"❌ Error in OperatorMode.execute: {e}")
            import traceback
            logger.error(f"📊 Full traceback: {traceback.format_exc()}")
            
            # Enhanced error context for vision_rag mode
            error_context = f"Execute error in {rag_mode} mode"
            if rag_mode == "vision_rag":
                error_context += " - This may be due to RAG database initialization or embedding model issues"
                logger.error("🔍 Vision RAG troubleshooting tips:")
                logger.error("  1. Check if embedding model is accessible")
                logger.error("  2. Verify RAG database files exist")
                logger.error("  3. Ensure sufficient GPU memory")
                logger.error("  4. Check OpenAI API key for GPT-4")
            
            error_message = f"{error_context}: {str(e)}"
            
            rag_data["execute_error"] = {
                "error": str(e),
                "error_type": type(e).__name__,
                "rag_mode": rag_mode,
                "screenshot_available": observation_VforD is not None,
                "traceback": traceback.format_exc()
            }
            
            return ("", error_message, "", {"action": "operator_wait", "action_input": "1000", "ms": 1000, "element_id": "execute_error_wait"}, [0, 0], rag_data)

    
    def _build_operator_messages(self, user_request: str, status_description: str,
                               previous_trace: str, feedback: str, 
                               rag_enabled: bool, rag_path: str, rag_mode: str = "description", rag_cache_dir: str = None) -> List[Dict[str, Any]]:
        """
        Build messages for OpenAI Operator
        
        Args:
            user_request: User's task request
            status_description: Current task status
            previous_trace: Previous action history
            feedback: Any feedback messages
            rag_enabled: Whether RAG is enabled
            rag_path: Path to RAG data
            rag_mode: RAG mode (description/vision/vision_rag)
            rag_cache_dir: RAG cache directory path
            
        Returns:
            Formatted messages for operator
        """
        if rag_enabled:
            try:
                # use cached RAG constructor to avoid duplicate initialization
                if rag_mode not in self.rag_constructors:
                    logger.info(f"🔄 Initializing RAG constructor for mode: {rag_mode}")
                    
                    # select RAG mode
                    if rag_mode == "vision":
                        from agent.Prompt.prompt_constructor import OperatorPromptVisionRetrievalConstructor
                        self.rag_constructors[rag_mode] = OperatorPromptVisionRetrievalConstructor()
                        constructor_type = "OperatorPromptVisionRetrievalConstructor"
                    elif rag_mode == "vision_rag":
                        from agent.Prompt.prompt_constructor import OperatorVisionRAGConstructor
                        self.rag_constructors[rag_mode] = OperatorVisionRAGConstructor()
                        constructor_type = "OperatorVisionRAGConstructor"
                    elif rag_mode == "description_rag":
                        from agent.Prompt.prompt_constructor import OperatorDescRAGConstructor
                        self.rag_constructors[rag_mode] = OperatorDescRAGConstructor()
                        constructor_type = "OperatorDescRAGConstructor"
                    else:  # rag_mode == "description"
                        from agent.Prompt.prompt_constructor import OperatorPromptDescriptionRetrievalConstructor
                        self.rag_constructors[rag_mode] = OperatorPromptDescriptionRetrievalConstructor()
                        constructor_type = "OperatorPromptDescriptionRetrievalConstructor"
                    
                    logger.info(f"✅ RAG constructor initialized and cached for mode: {rag_mode}")
                else:
                    logger.debug(f"🎯 Using cached RAG constructor for mode: {rag_mode}")
                    constructor_type = f"Operator{rag_mode.title().replace('_', '')}Constructor"
                
                rag_constructor = self.rag_constructors[rag_mode]
                
            except ImportError as import_error:
                logger.error(f"❌ Failed to import RAG constructor for mode {rag_mode}: {import_error}")
                logger.warning("🔄 Falling back to non-RAG mode")
                rag_enabled = False
                constructor_type = "fallback_non_rag"
            except Exception as rag_init_error:
                logger.error(f"❌ Failed to initialize RAG constructor for mode {rag_mode}: {rag_init_error}")
                logger.warning("🔄 Falling back to non-RAG mode")
                rag_enabled = False
                constructor_type = "fallback_non_rag"
            
            if hasattr(self, 'current_rag_data'):
                self.current_rag_data["rag_method"] = rag_constructor.__class__.__name__
                self.current_rag_data["rag_constructor_type"] = constructor_type
            
            previous_trace_list = []
            if previous_trace:
                trace_lines = previous_trace.split('\n')
                for line in trace_lines:
                    if line.strip():
                        # "Step X: thought -> action" format
                        if " -> " in line:
                            parts = line.split(" -> ")
                            if len(parts) >= 2:
                                thought_part = parts[0].strip()
                                action_part = parts[1].strip()
                                
                                # thought context（Get rid of "Step X: "）
                                if ":" in thought_part:
                                    thought = thought_part.split(":", 1)[1].strip()
                                else:
                                    thought = thought_part
                                
                                previous_trace_list.append({
                                    "thought": thought,
                                    "action": action_part,
                                    "reflection": ""
                                })
                        else:
                            previous_trace_list.append({
                                "thought": "Previous action",
                                "action": line.strip(),
                                "reflection": ""
                            })
            
            # RAG messages with error handling
            try:
                if constructor_type == "OperatorVisionRAGConstructor":
                    # OperatorVisionRAGConstructor rag_config parameter
                    messages = rag_constructor.construct(
                        user_request=user_request,
                        rag_path=rag_path,
                        previous_trace=previous_trace_list,
                        observation="",  # Operator not use DOM observation
                        feedback=feedback,
                        status_description=status_description,
                        screenshot_base64=self.current_screenshot,
                        rag_config=None,
                        rag_cache_dir=rag_cache_dir
                    )
                elif constructor_type == "OperatorDescRAGConstructor":
                    # OperatorDescRAGConstructor rag_config parameter
                    messages = rag_constructor.construct(
                        user_request=user_request,
                        rag_path=rag_path,
                        previous_trace=previous_trace_list,
                        observation="",  # Operator not use DOM observation
                        feedback=feedback,
                        status_description=status_description,
                        screenshot_base64=self.current_screenshot,
                        rag_config=None,
                        rag_cache_dir=rag_cache_dir
                    )
                else:
                    # other mode
                    messages = rag_constructor.construct(
                        user_request=user_request,
                        rag_path=rag_path,
                        previous_trace=previous_trace_list,
                        observation="",  # Operator not use DOM observation
                        feedback=feedback,
                        status_description=status_description,
                        screenshot_base64=self.current_screenshot
                    )
            except Exception as construct_error:
                logger.error(f"❌ RAG message construction failed: {construct_error}")
                logger.warning("🔄 Falling back to simple message construction")
                rag_enabled = False
                constructor_type = "construct_error_fallback"
            
            if hasattr(self, 'current_rag_data'):
                if constructor_type == "OperatorPromptVisionRetrievalConstructor":
                    # Vision RAG data
                    retrieved_images = []
                    if hasattr(rag_constructor, 'retrieved_image_paths') and rag_constructor.retrieved_image_paths:
                        for i, image_path_json in enumerate(rag_constructor.retrieved_image_paths):
                            try:
                                image_paths = json5.loads(image_path_json)
                                for path in image_paths:
                                    retrieved_images.append({
                                        "path": path,
                                        "task_index": i,
                                        "task_name": rag_constructor.retrieved_tasks[i] if i < len(rag_constructor.retrieved_tasks) else "Unknown"
                                    })
                            except Exception:
                                pass
                    
                    self.current_rag_data.update({
                        "rag_constructor_used": True,
                        "retrieved_tasks": getattr(rag_constructor, 'retrieved_tasks', []),
                        "retrieved_texts": getattr(rag_constructor, 'retrieved_texts', []),
                        "retrieved_images": retrieved_images,
                        "previous_trace_processed": previous_trace_list,
                        "messages_count": len(messages),
                        "user_request": user_request,
                        "status_description": status_description,
                        "feedback": feedback,
                        "current_screenshot": self.current_screenshot,
                        "screenshot_available": self.current_screenshot is not None
                    })
                elif constructor_type == "OperatorVisionRAGConstructor":
                    # Vision RAG with embedding-based retrieval data
                    retrieved_info = rag_constructor.get_last_retrieved_info()
                    
                    self.current_rag_data.update({
                        "rag_constructor_used": True,
                        "rag_type": "vision_embedding_retrieval",
                        "retrieved_info": retrieved_info,
                        "rag_db_initialized": rag_constructor.rag_db is not None,
                        "gpt4_available": rag_constructor.gpt4_client is not None,
                        "previous_trace_processed": previous_trace_list,
                        "messages_count": len(messages),
                        "user_request": user_request,
                        "status_description": status_description,
                        "feedback": feedback,
                        "current_screenshot": self.current_screenshot,
                        "screenshot_available": self.current_screenshot is not None
                    })
                    
                    # log retrieved info
                    if retrieved_info:
                        self.current_rag_data.update({
                            "retrieved_task_id": retrieved_info.get('cand_id', ''),
                            "retrieved_task_description": retrieved_info.get('task_description', ''),
                            "similarity_score": retrieved_info.get('score', 0.0),
                            "retrieved_context": retrieved_info.get('cand_text', '')[:200] + "..." if retrieved_info.get('cand_text') else ""
                    })
                elif constructor_type == "OperatorDescRAGConstructor":
                    # Description RAG with embedding-based retrieval data
                    retrieved_info = rag_constructor.get_last_retrieved_info()
                    
                    self.current_rag_data.update({
                        "rag_constructor_used": True,
                        "rag_type": "description_embedding_retrieval",
                        "retrieved_info": retrieved_info,
                        "rag_db_initialized": rag_constructor.rag_db is not None,
                        "gpt4_available": rag_constructor.gpt4_client is not None,
                        "previous_trace_processed": previous_trace_list,
                        "messages_count": len(messages),
                        "user_request": user_request,
                        "status_description": status_description,
                        "feedback": feedback,
                        "current_screenshot": self.current_screenshot,
                        "screenshot_available": self.current_screenshot is not None
                    })
                    
                    # log retrieved info
                    if retrieved_info:
                        self.current_rag_data.update({
                            "retrieved_task_id": retrieved_info.get('task_id', ''),
                            "retrieved_reference_preview": retrieved_info.get('retrieved_reference', ''),
                            "embedding_retrieval": retrieved_info.get('embedding_retrieval', False),
                            "gpt4_reranked": retrieved_info.get('gpt4_reranked', False)
                        })
                else:
                    # Description RAG
                    self.current_rag_data.update({
                        "rag_constructor_used": True,
                        "rag_reference": getattr(rag_constructor, 'reference', ""),
                        "previous_trace_processed": previous_trace_list,
                        "messages_count": len(messages),
                        "user_request": user_request,
                        "status_description": status_description,
                        "feedback": feedback,
                        "screenshot_available": self.current_screenshot is not None
                    })
                
                # log constructed prompt content
                safe_messages = []
                for msg in messages:
                    safe_msg = {"role": msg["role"]}
                    if msg["role"] == "system":
                        safe_msg["content"] = msg["content"]
                    elif msg["role"] == "user" and isinstance(msg["content"], list):
                        safe_content = []
                        for item in msg["content"]:
                            if item["type"] == "input_text":
                                safe_content.append({
                                    "type": "input_text",
                                    "text": item["text"][:500] + "..." if len(item["text"]) > 500 else item["text"]
                                })
                            elif item["type"] == "input_image":
                                safe_content.append({
                                    "type": "input_image",
                                    "image_url": "[base64_image_data_removed]"
                                })
                            elif item["type"] == "text":
                                safe_content.append({
                                    "type": "text",
                                    "text": item["text"][:1000] + "..." if len(item["text"]) > 1000 else item["text"]
                                })
                            elif item["type"] == "image_url":
                                safe_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": "[base64_image_data_removed_for_log_size]"}
                                })
                        safe_msg["content"] = safe_content
                    else:
                        safe_msg["content"] = str(msg["content"])[:500] + "..." if len(str(msg["content"])) > 500 else str(msg["content"])
                    safe_messages.append(safe_msg)
                
                self.current_rag_data["constructed_messages"] = safe_messages
            
            logger.info(f"🧠 RAG Constructor: {rag_constructor.__class__.__name__} (Mode: {rag_mode})")
            if rag_mode == "vision":
                logger.info(f"🖼️  Using Vision RAG with visual examples")
            elif rag_mode == "vision_rag":
                # For vision_rag mode, check if retrieval was successful
                if hasattr(rag_constructor, 'get_last_retrieved_info'):
                    retrieved_info = rag_constructor.get_last_retrieved_info()
                    if retrieved_info:
                        logger.info(f"🔍 Vision RAG: Successfully retrieved similar task (ID: {retrieved_info.get('cand_id', 'N/A')})")
                        logger.info(f"📊 Similarity Score: {retrieved_info.get('score', 0.0):.4f}")
                    else:
                        logger.info(f"⚠️  Vision RAG: No similar tasks retrieved")
                else:
                    logger.info(f"🤖 Vision RAG: Using embedding-based retrieval")
            elif rag_mode == "description_rag":
                if hasattr(rag_constructor, 'get_last_retrieved_info'):
                    retrieved_info = rag_constructor.get_last_retrieved_info()
                    if retrieved_info:
                        logger.info(f"🔍 Description RAG: Successfully retrieved similar task (ID: {retrieved_info.get('task_id', 'N/A')})")
                        logger.info(f"🧠 GPT-4 Re-ranking: {'Used' if retrieved_info.get('gpt4_reranked', False) else 'Not Used'}")
                        logger.info(f"📚 Reference Length: {len(retrieved_info.get('retrieved_reference', ''))}")
                    else:
                        logger.info(f"⚠️  Description RAG: No similar tasks retrieved")
                else:
                    logger.info(f"🤖 Description RAG: Using embedding-based retrieval")
            else: # description rag
                logger.info(f"📚 RAG Reference Length: {len(getattr(rag_constructor, 'reference', ''))}")
            
            return messages
        else:
            # Original simple message construction logic
            if hasattr(self, 'current_rag_data'):
                self.current_rag_data.update({
                    "rag_constructor_used": False,
                    "simple_message_construction": True
                })
            
            messages = []
            
            # System message for operator
            system_message = OperatorPrompts.operator_autonomous_simple_system
            
            if status_description:
                system_message += f"\n**Task Status**: {status_description}"
            
            if previous_trace:
                system_message += f"\n**Previous Actions**: {previous_trace}"
            
            if feedback:
                system_message += f"\n**Feedback**: {feedback}"
            
            messages.append({"role": "system", "content": system_message})
            
            # User message with task request (using template from operator_prompts.py)
            from jinja2 import Template
            user_message_template = Template(OperatorPrompts.operator_autonomous_user_template)
            user_message = user_message_template.render(user_request=user_request)
            
            messages.append({"role": "user", "content": user_message})
            
            return messages
    
    def _convert_operator_action(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenAI Operator action to compatible format for existing code
        
        Args:
            action_data: Raw action data from operator
            
        Returns:
            Compatible action format
        """
        action_type = action_data.get("type", "wait")
        
        if action_type == "click":
            # handle button field - keep button information but use operator_click
            button = action_data.get("button", "left")
            x_coord = action_data.get('x', 0)
            y_coord = action_data.get('y', 0)
            
            return {
                "action": "operator_click",
                "action_input": f"{x_coord},{y_coord}",
                "coordinates": [x_coord, y_coord],
                "button": button,
                "element_id": f"coord_{x_coord}_{y_coord}_{button}"
            }
        elif action_type == "double_click":
            button = action_data.get("button", "left")
            x_coord = action_data.get('x', 0)
            y_coord = action_data.get('y', 0)
            
            return {
                "action": "operator_double_click",
                "action_input": f"{x_coord},{y_coord}",
                "coordinates": [x_coord, y_coord],
                "button": button,
                "element_id": f"coord_{x_coord}_{y_coord}_double_{button}"
            }
        elif action_type == "type":
            return {
                "action": "operator_type",
                "action_input": action_data.get("text", ""),
                "text": action_data.get("text", ""),
                "element_id": "text_input"
            }
        elif action_type == "scroll":
            return {
                "action": "operator_scroll",
                "action_input": f"{action_data.get('scroll_x', 0)},{action_data.get('scroll_y', 0)}",
                "scroll_x": action_data.get('scroll_x', 0),
                "scroll_y": action_data.get('scroll_y', 0),
                "element_id": "scroll_action"
            }
        elif action_type == "keypress":
            keys = action_data.get("keys", [])
            return {
                "action": "operator_keypress",
                "action_input": ",".join(keys),
                "keys": keys,
                "element_id": "keypress_action"
            }
        elif action_type == "drag":
            path = action_data.get("path", [[0, 0], [0, 0]])
            return {
                "action": "operator_drag",
                "action_input": f"{path[0][0]},{path[0][1]}-{path[-1][0]},{path[-1][1]}",
                "path": path,
                "element_id": f"drag_{path[0][0]}_{path[0][1]}_to_{path[-1][0]}_{path[-1][1]}"
            }
        elif action_type == "wait":
            return {
                "action": "operator_wait",
                "action_input": str(action_data.get("ms", 1000)),
                "ms": action_data.get("ms", 1000),
                "element_id": "wait_action"
            }
        elif action_type == "get_final_answer" or action_type == "final_answer":
            return {
                "action": "get_final_answer",
                "action_input": action_data.get("answer", action_data.get("text", "")),
                "element_id": "final_answer"
            }
        else:
            return {
                "action": "operator_wait",
                "action_input": "1000",
                "ms": 1000,
                "element_id": "unknown_action_wait"
            }
    
    def _parse_action_from_text(self, text_response: str) -> Dict[str, Any]:
        """
        Attempt to parse an action from a text response.
        This handles cases where the operator returns action info in text_response instead of actions array.
        """
        try:
            # Try to find JSON in the text response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                action_data = json.loads(json_str)
                
                # Extract coordinates and action type
                action_type = action_data.get("action", "click")
                action_input = action_data.get("action_input", {})
                
                if action_type == "click" and isinstance(action_input, dict):
                    x = action_input.get("x", 0)
                    y = action_input.get("y", 0)
                    return {
                        "action": "operator_click",
                        "action_input": f"{x},{y}",
                        "coordinates": [x, y],
                        "element_id": f"coord_{x}_{y}"
                    }
                elif action_type == "type" and isinstance(action_input, dict):
                    text = action_input.get("text", "")
                    return {
                        "action": "operator_type",
                        "action_input": text,
                        "text": text,
                        "element_id": "text_input"
                    }
                # Add more action types as needed
                
        except Exception as e:
            logger.warning(f"Could not parse JSON from text_response: {e}")
        
        # Fallback to simple text analysis
        text_lower = text_response.lower()
        if "click" in text_lower:
            # Try to extract coordinates from text
            coord_match = re.search(r'"x":\s*(\d+).*?"y":\s*(\d+)', text_response)
            if coord_match:
                x, y = int(coord_match.group(1)), int(coord_match.group(2))
                return {
                    "action": "operator_click",
                    "action_input": f"{x},{y}",
                    "coordinates": [x, y],
                    "element_id": f"coord_{x}_{y}"
                }
            return {"action": "operator_click", "action_input": "640,360", "coordinates": [640, 360], "element_id": "default_click"}
        elif "type" in text_lower:
            return {"action": "operator_type", "action_input": "", "text": "", "element_id": "type_action"}
        elif "scroll" in text_lower:
            return {"action": "operator_scroll", "action_input": "0,100", "scroll_x": 0, "scroll_y": 100, "element_id": "scroll_action"}
        elif "wait" in text_lower:
            return {"action": "operator_wait", "action_input": "1000", "ms": 1000, "element_id": "wait_action"}
        elif "final_answer" in text_lower or "complete" in text_lower or "finished" in text_lower:
            # Extract answer content if available
            answer_content = ""
            answer_match = re.search(r'(?:answer|result|content)[:：]\s*"([^"]*)"', text_response, re.IGNORECASE)
            if answer_match:
                answer_content = answer_match.group(1)
            elif "task complete" in text_lower:
                answer_content = "Task completed successfully"
            
            return {
                "action": "get_final_answer",
                "action_input": answer_content,
                "element_id": "final_answer"
            }
        else:
            return {"action": "operator_wait", "action_input": "1000", "ms": 1000, "element_id": "text_parse_fallback_wait"}
    
    def _load_rag_context(self, rag_path: str) -> str:
        """
        Load RAG context from file
        
        Args:
            rag_path: Path to RAG data file
            
        Returns:
            RAG context string
        """
        try:
            if os.path.exists(rag_path):
                with open(rag_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading RAG context: {e}")
        return ""
    
    def reset_conversation(self):
        """Reset the conversation history and optionally clear RAG constructors"""
        self.conversation_history.clear()
        self.current_screenshot = None
        if hasattr(self.operator_model, 'reset_conversation'):
            self.operator_model.reset_conversation()
    
    def clear_rag_cache(self):
        """Clear cached RAG constructors to free memory"""
        logger.info(f"🧹 Clearing RAG constructor cache ({len(self.rag_constructors)} constructors)")
        self.rag_constructors.clear()
    
    def get_rag_cache_status(self):
        """Get status of RAG constructor cache"""
        return {
            "cached_modes": list(self.rag_constructors.keys()),
            "cache_count": len(self.rag_constructors)
        }
