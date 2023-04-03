import tensorflow
import base64
import requests
import os

class SaveModelToGitHub(tensorflow.keras.callbacks.Callback):
    def __init__(self, github_settings, name="model", only_save_at_end=False, delete_after_upload=False):
        super().__init__()
        self.github_settings = github_settings
        self.name = name
        self.only_save_at_end = only_save_at_end
        self.delete_after_upload = delete_after_upload

        github_check_fields = ["access_token", "owner", "repository"]
        for field in github_check_fields:
            if field not in github_settings or github_settings[field].strip() == "":
                raise AttributeError(
                    f"{field} field is not in github settings or is empty")

    def on_epoch_end(self, epoch, logs=None):
        if self.only_save_at_end == False or (self.only_save_at_end and self.params["epochs"] - 1 == epoch):
            self._save_model(epoch)

    def _save_model(self, epoch):
        filename = f"{self.name}-epoch-{epoch + 1}.h5"
        filepath = f"./{filename}"
        self.model.save(filepath)
        github_settings = self.github_settings

        with open(filepath, "rb") as file:
            content = file.read()

        content_base64 = base64.b64encode(content).decode()
        data = {
            'message': f"Back up for {self.name} epoch {epoch + 1}",
            'content': content_base64
        }
        headers = {
            'Authorization': f'token {github_settings["access_token"]}',
            'Content-Type': 'application/json'
        }

        owner = github_settings["owner"]
        repo = github_settings["repository"]
        folder = github_settings["folder"] if "folder" in github_settings else "ai-models"
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}/{filename}"
        resp = requests.put(url, headers=headers, json=data)

        if resp.ok == False:
            print(resp.content)
            raise RuntimeError("Commit couldn't be made to github")

        if self.delete_after_upload:
            os.remove(filepath)
