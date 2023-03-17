import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.join("..", ".."))),
)

from ie_utils_tool import Mattermost

from exp.bert.train import train
from exp.tools.conf import CHANNEL_ID, TOKEN, URL

p = argparse.ArgumentParser(description="training bert model...")
p.add_argument("--export_name")
p.add_argument("--use_aug")
args = p.parse_args()

export_name = args.export_name
use_aug = args.use_aug
if use_aug == "True":
    use_aug = True
else:
    use_aug = False
cptk_path, log_path = train(use_aug)

mattermost = Mattermost(url=URL, token=TOKEN)
abs_log_path = os.path.abspath(log_path)
abs_cptk_path = os.path.abspath(cptk_path)
message = f"""Training is just finished.
To download the training results, you can execute the following command.

```
rsync -avhz "braun:{abs_log_path}" ./{export_name}
rsync -avhz "braun:{abs_cptk_path}" ./{export_name}
```
"""
mattermost.send_message(CHANNEL_ID, message)
