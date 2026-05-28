# HireFlow AI - Replication Package

## Paper Information
**Title:** HireFlow AI: An Intelligent Framework for Automated Talent Acquisition Using Generative AI and Workflow Orchestration

**Authors:** Shaik Abdul Razak, Shiva Kumar Kasula, Rajesh Potharla, Sirikonda Sreeja, Suraj Bhan Kumar, Narsaiah Domala, Edukondalu Simhadati, Ramagiri Praveen Kumar

## Quick Start

### Prerequisites
1. n8n (self-hosted or cloud) - https://n8n.io
2. Google Gemini API key - https://makersuite.google.com/app/apikey
3. Firebase project - https://console.firebase.com
4. Gmail account for sending emails

### Setup Instructions

#### 1. Import n8n Workflow
1. Open n8n dashboard.
2. Click **Import** → **From File**.
3. Select `workflows/hireflow-workflow.json`.
4. Click **Execute Workflow**.

#### 2. Configure Credentials in n8n
- **Google Gemini API:** Add API key in credential settings.
- **Gmail OAuth:** Connect your Gmail account.
- **Firebase:** Add service account credentials.

#### 3. Run Evaluation
```bash
cd evaluation
pip install -r requirements.txt
python evaluate.py --data ../sample-data/resumes/
```

### Repository Contents
| File/Directory | Description |
| :--- | :--- |
| `workflows/hireflow-workflow.json` | Complete n8n workflow export with 8 nodes |
| `evaluation/evaluate.py` | Python script for accuracy, F1, DPD metrics |
| `prompts/gemini-prompts.txt` | Prompt templates for Gemini (parsing, ranking, email) |
| `sample-data/resumes/` | Anonymized sample resumes in JSON format for testing |
| `results/sample-output.json` | Example output from running the workflow |

### Expected Output
After running the workflow or evaluation, you will get:
- Ranked list of candidates with match scores (0-1).
- Personalized interview invitation emails.
- Analytics stored in Firebase.

### Citation
If you use this code, please cite:
```bibtex
@article{razak2024hireflow,
  title={HireFlow AI: An Intelligent Framework for Automated Talent Acquisition Using Generative AI and Workflow Orchestration},
  author={Razak, Shaik Abdul and Kasula, Shiva Kumar and Potharla, Rajesh and Sreeja, Sirikonda and Kumar, Suraj Bhan and Domala, Narsaiah and Simhadati, Edukondalu and Kumar, Ramagiri Praveen},
  journal={Journal of the Brazilian Computer Society},
  year={2024}
}
```

### License
MIT License

### Contact
Corresponding Author: Dr. R. Praveen Kumar (ramagiri.praveen594@gmail.com)
