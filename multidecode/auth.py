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
        print('Using HUGGINGFACE environment variable for authentication.')
        try:
            import dotenv
            dotenv.load_dotenv()
        except:
            pass
        token = os.getenv("HUGGINGFACE")
    if token is None:
        raise ValueError("No Hugging Face token provided.")
    login(token=token)
