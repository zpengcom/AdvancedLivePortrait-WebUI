from enum import Enum
from gradio_i18n import Translate, gettext as _


class ModelType(Enum):
    HUMAN = _("Human")
    ANIMAL = _("Animal")


REPO_MARKDOWN = """
## [AdvancedLivePortrait-WebUI](https://github.com/jhj0517/AdvancedLivePortrait-WebUI/tree/master)
"""

GRADIO_CSS = """
#md_project a {
  color: black;
  text-decoration: none;
}
#md_project a:hover {
  text-decoration: underline;
}
"""