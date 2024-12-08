import gdown
import os


def download():
    gdown.download("https://drive.google.com/uc?id=1KWTTiJRy6NjNM0Jw7A2gXgb9TdlDqXu0")

    os.mkdir('to_download')
    os.rename("model_best_hifi.pth", "to_download/model_best_hifi.pth")


if __name__ == "__main__":
    download()
