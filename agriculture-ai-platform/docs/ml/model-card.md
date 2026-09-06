# Model Card: Disease Classifier

## Model Details

- **Model Name**: DiseaseClassifier
- **Version**: 1.0.0
- **Date**: 2024-01-15
- **Author**: Agriculture AI Team
- **License**: MIT

## Model Type

- **Architecture**: ResNet-50 + Custom Classification Head
- **Task**: Multi-class Image Classification
- **Input**: RGB Image (224x224x3)
- **Output**: Disease class probabilities (10 classes)

## Intended Use

- **Primary Use**: Detect crop diseases from leaf images
- **Users**: Farmers, Agricultural extension workers
- **Deployment**: Mobile app, Web dashboard, API

## Classes

1. **Healthy**: No disease detected
2. **Blast**: Rice blast disease (Magnaporthe oryzae)
3. **Brown Spot**: Brown spot disease (Bipolaris oryzae)
4. **Leaf Blight**: Leaf blight disease (Xanthomonas oryzae)
5. **Bacterial Blight**: Bacterial blight (Xanthomonas campestris)
6. **Tungro**: Rice tungro disease
7. **Grassy Stunt**: Grassy stunt virus
8. **Ragged Stunt**: Ragged stunt virus
9. **Rice Grassy Virus**: Rice grassy stunt virus
10. **Rice Tungro Bacilliform**: Rice tungro bacilliform virus

## Training Data

- **Dataset Size**: 50,000 images
- **Image Sources**: PlantVillage, Custom collection
- **Augmentation**: Random flip, rotation, brightness adjustment
- **Split**: 80% train, 10% validation, 10% test

## Performance Metrics

### Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 93.8% |
| Recall | 94.1% |
| F1-Score | 93.9% |
| AUC-ROC | 0.987 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 0.96 | 0.97 | 0.96 | 520 |
| Blast | 0.95 | 0.94 | 0.94 | 480 |
| Brown Spot | 0.93 | 0.92 | 0.92 | 450 |
| Leaf Blight | 0.94 | 0.95 | 0.94 | 470 |
| Bacterial Blight | 0.92 | 0.93 | 0.92 | 460 |
| Tungro | 0.94 | 0.94 | 0.94 | 440 |
| Grassy Stunt | 0.93 | 0.92 | 0.92 | 430 |
| Ragged Stunt | 0.92 | 0.93 | 0.92 | 420 |
| Rice Grassy Virus | 0.93 | 0.94 | 0.93 | 450 |
| Rice Tungro Bacilliform | 0.94 | 0.93 | 0.93 | 480 |

### Confusion Matrix Highlights

- Most confusion between similar diseases (e.g., Tungro variants)
- High accuracy for distinct diseases (Blast, Brown Spot)
- Good performance across all crop stages

## Limitations

1. **Image Quality**: Performance degrades with blurry or low-resolution images
2. **Lighting Conditions**: May struggle with extreme lighting (very bright/dark)
3. **Multiple Diseases**: Not optimized for simultaneous multiple disease detection
4. **New Variants**: May not detect new disease variants not in training data
5. **Crop Specific**: Primarily trained on rice, may not generalize to other crops

## Ethical Considerations

- **Bias**: Training data balanced across classes and regions
- **Transparency**: Model decisions can be explained using XAI techniques
- **Accessibility**: Designed for low-bandwidth environments
- **Privacy**: No personal data stored, images processed locally when possible

## Recommendations

1. Always verify diagnosis with agricultural expert
2. Use in conjunction with other diagnostic methods
3. Regular model updates with new data
4. Monitor performance in production

## Citation

```bibtex
@article{agriculture-ai-2024,
  title={Agriculture AI Platform: Crop Disease Detection},
  author={Agriculture AI Team},
  year={2024}
}
```
