from enum import Enum
from gradio_i18n import Translate, gettext as _


class ModelType(Enum):
    HUMAN = _("Human")
    ANIMAL = _("Animal")


class SamplePart(Enum):
    ONLY_EXPRESSION = _("OnlyExpression")
    ONLY_ROTATION = _("OnlyRotation")
    ONLY_MOUTH = _("OnlyMouth")
    ONLY_EYES = _("OnlyEyes")
    ALL = _("All")


REPO_MARKDOWN = """
## [AdvancedLivePortrait-WebUI](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/tree/master)

The space runs on the CPU, so it's slow. If you want faster inference, try running it locally using the link above. 
"""

GRADIO_CSS = """
#md_project a {
  color: black;
  text-decoration: none;
}
#md_project a:hover {
  text-decoration: underline;
}

#blink_slider .md.svelte-7ddecg.chatbot.prose {
    font-size: 0.7em; 
}
"""