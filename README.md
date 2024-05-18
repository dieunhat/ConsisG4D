# ConsisG4D

Install required packages by running the following command:
```bash
pip3 install -r requirements.txt
``` 

Wandb setup:
```bash
wandb login
```
Enter the API key when prompted and replace the 'entity' in the `wandb.init()` function with your username (in `main.py`).

To start training, execute the following command:
```bash
python3 main.py
```
