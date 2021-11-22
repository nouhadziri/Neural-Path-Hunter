import json
import os

try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False


def is_wandb_available() -> bool:
    return _wandb_available


def authorize_wandb(overwrite_apikey: bool = False) -> bool:
    assert is_wandb_available()

    authorized = False
    if overwrite_apikey or not os.getenv("WANDB_API_KEY", None):
        if os.path.isfile("./settings.json"):
            with open("./settings.json") as f:
                data = json.load(f)

            os.environ["WANDB_API_KEY"] = data.get("wandbapikey")
            authorized = True
    else:
        authorized = True

    return authorized
