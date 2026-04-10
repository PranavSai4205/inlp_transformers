# INLP Project: Evasion and Stance Detection Pipeline

This is the final repository for the INLP sequence model project for Evasion and Stance detection. It contains the implementation of a custom `MultiTaskV3C` model built on top of `distilbert-base-uncased`, trained via a multi-task learning setup using focal and supervised contrastive loss mechanisms.

## Repository Structure

- `ablation/`: Contains the ablation study code developed post mid-submission. It holds different model variants to evaluate the effectiveness of individual architectural modifications (e.g., contrastive loss, gating layers, different focal loss configurations).
- `final/`: Contains the final consolidated code for submission. This includes the finalized model architecture (`model.py`), the robust dataset processing utilities (`data.py`), hyperparameter configs (`config.py`), and the training script (`train.py`).
- `inference.py`: The inference code for the final model. It consumes the model weights saved after running the training script in the `final` branch (typically `best_model_v3c_final.pt`).
- `notebooks/`: Contains the IPython notebooks used for local exploratory data analysis and baseline tests.

## Model Architecture

Our custom `MultiTaskV3C` configuration introduces multiple enhancements:
1. **Token-Level Gating**: Question representations gate answer tokens for dynamic filtration.
2. **Token-Level Cross-Attention**: Incorporates explicit question-answer cross-attention mechanisms mapping tokens to relevant explanations.
3. **Focal Loss**: Heavily penalizes easy samples while focusing training on the hardest cases using gamma=3.0.
4. **Contrastive Loss (NT-Xent)**: Supervised contrastive objective pulling similar labels together.
5. **Answer Length Features**: Passing explicit $\log(\text{len})$ feature into the classification heads.

## Usage

### 1. Training the Final Model

To train the final Multi-Task model, ensure data is present in the specified directories in `final/config.py` and run:

```bash
cd final
python train.py
```
This script will produce `output/best_model_v3c_final.pt`.

### 2. Running Inference

To run the inference using the trained script, execute `inference.py` from the root directory:

**Interactive Demo Mode:**
Allows you to directly input Question and Answer subsets. It predicts evasion probability, stance, and pulls out the most explanatory clause.
```bash
python inference.py
```

**Dataset Demo Evaluation Mode:**
Runs the inference routine automatically on a small 5-example validation subset:
```bash
python inference.py --demo
```

**Batch CSV Processing:**
Pass a `.csv` file containing `question` and `answer` columns.
```bash
python inference.py --input raw_test_data.csv --output predictions.csv
```

## Dependencies
- PyTorch
- Transformers (HuggingFace)
- Pandas, Numpy, tqdm
