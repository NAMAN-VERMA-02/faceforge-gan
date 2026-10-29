# Runs the Gradio demo anywhere Docker runs (Render, Cloud Run, a VM...).
#   docker build -t faceforge .
#   docker run -p 7860:7860 -e FACEFORGE_MODEL=<hf-user>/faceforge-gan faceforge
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy huggingface_hub "gradio>=4.44"

# Copies the code plus whichever trained model folders exist (model/, model_quick/).
# Everything else is excluded by .dockerignore.
COPY . .

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
