# [CMPB 2024] Empirical Validation of Conformal Prediction for Trustworthy Skin Lesions Classification

## Abstract
**Background and objective:** Uncertainty quantification is a pivotal field that contributes to realizing reliable and robust systems. It becomes instrumental in fortifying safe decisions by providing complementary information, particularly within high-risk applications. Existing studies have explored various methods that often operate under specific assumptions or necessitate substantial modifications to the network architecture to effectively account for uncertainties. The objective of this paper is to study Conformal Prediction, an emerging distribution-free uncertainty quantification technique, and provide a comprehensive understanding of the advantages and limitations inherent in various methods within the medical imaging field.

**Methods:** In this study, we developed Conformal Prediction, Monte Carlo Dropout, and Evidential Deep Learning approaches to assess uncertainty quantification in deep neural networks. The effectiveness of these methods is evaluated using three public medical imaging datasets focused on detecting pigmented skin lesions and blood cell types.

**Results:** The experimental results demonstrate a significant enhancement in uncertainty quantification with the utilization of the Conformal Prediction method, surpassing the performance of the other two methods. Furthermore, the results present insights into the effectiveness of each uncertainty method in handling Out-of-Distribution samples from domain-shifted datasets. Our code is available at:

**Conclusions:** Our conclusion highlights a robust and consistent performance of conformal prediction across diverse testing conditions. This positions it as the preferred choice for decision-making in safety-critical applications.

## Usage
To configure the environment, see the `requirements.txt` file. You can install the necessary dependencies with the following command:
```bash
pip install -r requirements.txt

# Datasets
1) [HAM10000 (HAM) dataset](https://www.nature.com/articles/sdata2018161)
2) [Dermofit (DMF) dataset](https://licensing.edinburgh-innovations.ed.ac.uk/product/dermofit-image-library)
