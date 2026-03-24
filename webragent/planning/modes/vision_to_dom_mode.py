import json5
import logging

from agent.Prompt import *
from agent.LLM import *
from .base_mode import InteractionMode
from ..action_parser import ActionParser

logger = logging.getLogger(__name__)


class VisionToDomMode(InteractionMode):
    def __init__(self, text_model=None, visual_model=None):
        super().__init__(text_model, visual_model)

    async def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        vision_act_request = ObservationVisionActPromptConstructor().construct(
            user_request, previous_trace, observation_VforD, feedback, status_description)
        max_retries = 3
        for attempt in range(max_retries):
            vision_act_response, error_message = await self.visual_model.request(vision_act_request)
            # Blue output
            print(f"\033[36mvision_act_response:\n{vision_act_response}")
            print("\033[0m")  # Reset color
            planning_response_thought, planning_response_get = ActionParser().extract_thought_and_action(
                vision_act_response)
            actions = {
                'goto': "Found 'goto' in the vision_act_response.",
                # 'google_search': "Found 'google_search' in the vision_act_response.",
                'switch_tab': "Found 'switch_tab' in the vision_act_response.",
                'scroll_down': "Found 'scroll_down' in the vision_act_response.",
                'scroll_up': "Found 'scroll_up' in the vision_act_response.",
                'go_back': "Found 'go_back' in the vision_act_response."
            }
            # Check if the action is in the predefined action list
            actions_found = False
            for action, message in actions.items():
                if action == planning_response_get.get('action'):
                    print(message)
                    actions_found = True
                    # The action does not need to be changed
                    # `target_element` should not exist, if it does, it's not used
                    break

            if not actions_found:
                # print("None of 'goto', 'google_search', 'switch_tab', 'scroll_down', 'scroll_up', or 'go_back' were found in the vision_act_response.")
                print("None of 'goto', 'switch_tab', 'scroll_down', 'scroll_up', or 'go_back' were found in the vision_act_response.")

                target_element = planning_response_get.get('target_element')
                description = planning_response_get.get('description')

                # If the target element is None or does not exist
                if not target_element:
                    print("The 'target_element' is None or empty.")
                    continue

                # Construct the request from vision to DOM
                planning_request = VisionToDomPromptConstructor().construct(target_element, description,
                                                                            observation)
                print(f"\033[35mplanning_request:{planning_request}")
                print("\033[0m")

                # Send the request and wait for the response
                planning_response_dom, error_message = await self.text_model.request(planning_request)
                print(
                    f"\033[34mVisionToDomplanning_response:\n{planning_response_dom}")
                print("\033[0m")
                # Parse the element ID
                element_id = ActionParser().get_element_id(planning_response_dom)
                if element_id == "-1":
                    print("The 'element_id' is not found in the planning_response.")
                    continue  # If the 'element_id' is not found, continue to the next iteration of the loop
                else:
                    planning_response_get['element_id'] = element_id
                    break  # If the 'element_id' is found, break the loop

            else:
                # If a predefined action is found, there is no need to retry, exit the loop directly
                break

        planning_response_json_str = json5.dumps(
            planning_response_get, indent=2)
        planning_response = f'```\n{planning_response_json_str}\n```'
        # Check if the maximum number of retries has been reached
        if attempt == max_retries - 1:
            print("Max retries of vision_act reached. Unable to proceed.")

        return planning_response, error_message, planning_response_thought, planning_response_get
