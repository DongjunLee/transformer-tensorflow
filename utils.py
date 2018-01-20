import json
import os.path

from hbconfig import Config
import requests



def get_rev_vocab(vocab):
    if vocab is None:
        return None
    return {idx: key for key, idx in vocab.items()}


def send_message_to_slack(config_name):
    project_name = os.path.basename(os.path.abspath("."))

    data = {
        "text": f"The learning is finished with *{project_name}* Project using `{config_name}` config."
    }

    webhook_url = Config.slack.webhook_url
    if webhook_url == "":
        print(data["text"])
    else:
        requests.post(Config.slack.webhook_url, data=json.dumps(data))
