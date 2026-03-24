import logging

logger = logging.getLogger(__name__)


class InteractionMode:
    def __init__(self, text_model=None, visual_model=None):
        self.text_model = text_model
        self.visual_model = visual_model

    def execute(self, status_description, user_request, previous_trace, observation, feedback, observation_VforD):
        # Returns a six-tuple containing None, consistent with DomMode
        return None, None, None, None, None, None
