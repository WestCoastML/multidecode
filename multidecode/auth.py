from huggingface_hub import login
import os

def hf_login(token: str | None = None, from_env: bool = True, colab_userdata: bool = True):
    if token is None and colab_userdata:
        try:
            from google.colab import userdata
            token = userdata.get('huggingface')
        except ImportError:
            pass
    if token is None and from_env:
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token is None:
        raise ValueError("No Hugging Face token provided.")
    login(token=token)