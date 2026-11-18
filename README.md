# Fast.ai image classifier on Streamlit

## Repository files

Keep `app.py`, `requirements.txt`, and your exported `model.pkl` in the
same repository folder.

## Export your trained model

In the training notebook, run this after `learn.fine_tune(...)` finishes:

```python
learn.export("model.pkl")
```

Download `model.pkl` from the notebook environment and upload it to this
repository. It must be a Fast.ai exported learner.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the model once and accepts image uploads for prediction.

Security note: only load a `.pkl` file that you created or trust. Python
pickle files can execute code when loaded.
